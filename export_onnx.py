"""
SAM3 ONNX Export Script (Dynamic Batch + TensorRT Compatible)
Adapted for fine-tuned checkpoints

Usage:
    # Export all modules from fine-tuned checkpoint:
    python export_onnx.py --all --checkpoint outputs/test_checkpoint.pt --output-dir onnx-models --device cuda
    
    # Export only vision encoder:
    python export_onnx.py --module vision --checkpoint outputs/test_checkpoint.pt --output-dir onnx-models
    
    # Export from base model (no fine-tuning):
    python export_onnx.py --all --output-dir onnx-models
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

# Import from SAM3 package
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image import Sam3Image


# TensorRT-compatible Position Encoding
def compute_sine_position_encoding(
    shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    scale: float = 2 * math.pi,
) -> torch.Tensor:
    """Compute sine position encoding (using arange instead of cumsum for TensorRT compatibility)"""
    batch_size, channels, height, width = shape

    y_embed = (
        torch.arange(1, height + 1, dtype=dtype, device=device)
        .view(1, height, 1)
        .expand(batch_size, height, width)
    )
    x_embed = (
        torch.arange(1, width + 1, dtype=dtype, device=device)
        .view(1, 1, width)
        .expand(batch_size, height, width)
    )

    eps = 1e-6
    y_embed = y_embed / (height + eps) * scale
    x_embed = x_embed / (width + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)

    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class VisionEncoderWrapper(nn.Module):
    """Vision Encoder: ViT backbone + FPN neck
    
    SAM3 structure:
    - model.backbone.vision_backbone: Sam3DualViTDetNeck (contains ViT trunk + FPN convs)
    - model.backbone.vision_backbone.trunk: ViT backbone
    - model.backbone.vision_backbone.convs: FPN convolutions
    - model.backbone.vision_backbone.position_encoding: Position encoding
    """

    def __init__(self, sam3_model: Sam3Image, device="cpu", image_size=1008):
        super().__init__()
        
        # Get the vision backbone from SAM3
        self.vision_backbone = sam3_model.backbone.vision_backbone
        
        # Pre-compute FPN sine position encoding for level 2 (72x72)
        # d_model is typically 256
        num_pos_feats = 128  # 256 // 2
        pos_enc_2 = compute_sine_position_encoding(
            shape=(1, 256, 72, 72),
            device=device,
            dtype=torch.float32,
            num_pos_feats=num_pos_feats,
        )
        self.register_buffer("pos_enc_2", pos_enc_2)

    def forward(self, images: torch.Tensor):
        """Forward pass through vision encoder.
        
        Args:
            images: Input images [B, 3, 1008, 1008]
            
        Returns:
            fpn_feat_0: [B, 256, 288, 288] - highest resolution FPN features
            fpn_feat_1: [B, 256, 144, 144] - mid resolution FPN features  
            fpn_feat_2: [B, 256, 72, 72] - lowest resolution FPN features
            fpn_pos_2: [B, 256, 72, 72] - position encoding for level 2
        """
        batch_size = images.shape[0]
        
        # Forward through vision backbone (ViT + FPN neck)
        # Returns: sam3_features, sam3_pos, sam2_features, sam2_pos
        fpn_features, fpn_pos, _, _ = self.vision_backbone(images)
        
        # fpn_features is a list of feature maps at different scales
        # Typically: [4x, 2x, 1x, 0.5x] scale factors
        # We return the first 3 levels (skip the 0.5x level)
        return (
            fpn_features[0],  # [B, 256, 288, 288] - 4x upsampled
            fpn_features[1],  # [B, 256, 144, 144] - 2x upsampled
            fpn_features[2],  # [B, 256, 72, 72] - 1x (native)
            self.pos_enc_2.expand(batch_size, -1, -1, -1),  # [B, 256, 72, 72]
        )


class TextEncoderONNXWrapper(nn.Module):
    """ONNX-compatible Text Encoder that takes pre-tokenized inputs.
    
    For ONNX export, we need to export just the transformer part,
    as tokenization must happen outside the model.
    """

    def __init__(self, sam3_model: Sam3Image):
        super().__init__()
        # Get the text encoder components
        lang_backbone = sam3_model.backbone.language_backbone
        self.text_encoder = lang_backbone.text_encoder
        self.text_projection = lang_backbone.text_projection

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass through text encoder with pre-tokenized inputs.
        
        Args:
            input_ids: [B, seq_len] - tokenized input IDs
            attention_mask: [B, seq_len] - attention mask
            
        Returns:
            text_features: [B, seq_len, hidden_dim] - projected text features
            text_mask: [B, seq_len] - boolean attention mask
        """
        # Forward through CLIP text encoder
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get hidden states and project
        text_features = text_outputs.last_hidden_state
        text_features = self.text_projection(text_features)
        text_mask = attention_mask > 0
        
        return text_features, text_mask


class GeometryEncoderWrapper(nn.Module):
    """Geometry Encoder for box/point prompts.
    
    SAM3 structure:
    - model.geometry_encoder: SequenceGeometryEncoder
    """

    def __init__(self, sam3_model: Sam3Image):
        super().__init__()
        self.geometry_encoder = sam3_model.geometry_encoder

    def forward(
        self,
        input_boxes: torch.Tensor,
        input_boxes_labels: torch.Tensor,
        fpn_feat: torch.Tensor,
        fpn_pos: torch.Tensor,
    ):
        """Forward pass through geometry encoder.
        
        Args:
            input_boxes: [B, num_boxes, 4] - boxes in cxcywh format (normalized 0-1)
            input_boxes_labels: [B, num_boxes] - box labels (0=background, 1=foreground, -10=padding)
            fpn_feat: [B, 256, 72, 72] - FPN features at level 2
            fpn_pos: [B, 256, 72, 72] - position encoding for level 2
            
        Returns:
            geometry_features: [B, num_prompts, hidden_dim] - encoded geometry prompts
            geometry_mask: [B, num_prompts] - attention mask
        """
        batch_size = input_boxes.shape[0]
        device = input_boxes.device
        
        # Build prompt dict for geometry encoder
        prompt_dict = {
            "boxes": input_boxes,
            "box_labels": input_boxes_labels,
        }
        
        # Build vision features dict
        vision_dict = {
            "vision_features": fpn_feat,
            "vision_pos_enc": [fpn_pos],
        }
        
        # Forward through geometry encoder
        # Returns encoded prompts and mask
        geometry_output = self.geometry_encoder(
            prompts=prompt_dict,
            vision_features=vision_dict,
        )
        
        return geometry_output["prompt_features"], geometry_output["prompt_mask"]


class DecoderWrapper(nn.Module):
    """Full decoder: Transformer Encoder + Decoder + Segmentation Head.
    
    SAM3 structure:
    - model.transformer: TransformerWrapper (encoder + decoder)
    - model.segmentation_head: UniversalSegmentationHead
    - model.dot_prod_scoring: DotProductScoring
    """

    def __init__(self, sam3_model: Sam3Image):
        super().__init__()
        self.transformer = sam3_model.transformer
        self.segmentation_head = sam3_model.segmentation_head
        self.dot_prod_scoring = sam3_model.dot_prod_scoring

    def forward(
        self,
        fpn_feat_0: torch.Tensor,
        fpn_feat_1: torch.Tensor,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        """Forward pass through decoder.
        
        Args:
            fpn_feat_0: [B, 256, 288, 288] - FPN features level 0
            fpn_feat_1: [B, 256, 144, 144] - FPN features level 1
            fpn_feat_2: [B, 256, 72, 72] - FPN features level 2
            fpn_pos_2: [B, 256, 72, 72] - position encoding for level 2
            prompt_features: [B, num_prompts, hidden_dim] - encoded prompts
            prompt_mask: [B, num_prompts] - prompt attention mask
            
        Returns:
            pred_masks: [B, num_queries, H, W] - predicted masks
            pred_boxes: [B, num_queries, 4] - predicted boxes (xyxy format)
            pred_logits: [B, num_queries] - classification logits
            presence_logits: [B, num_queries] - presence/confidence logits
        """
        batch_size = fpn_feat_0.shape[0]
        
        # Build vision features structure
        backbone_fpn = [fpn_feat_0, fpn_feat_1, fpn_feat_2]
        vision_pos_enc = [fpn_pos_2]  # Only need pos for level 2
        
        # Forward through transformer encoder
        encoder_output = self.transformer.encoder(
            vision_features=[fpn_feat_2],
            text_features=prompt_features,
            vision_pos_embeds=vision_pos_enc,
            text_mask=prompt_mask,
        )
        
        # Forward through transformer decoder
        decoder_output = self.transformer.decoder(
            vision_features=encoder_output.last_hidden_state,
            text_features=encoder_output.text_features,
            vision_pos_encoding=encoder_output.pos_embeds_flattened,
            text_mask=prompt_mask,
            spatial_shapes=encoder_output.spatial_shapes,
        )
        
        # Get box predictions
        box_head = self.transformer.decoder.box_head
        all_box_offsets = box_head(decoder_output.intermediate_hidden_states)
        reference_boxes_inv_sig = self._inverse_sigmoid(decoder_output.reference_boxes)
        all_pred_boxes = self._box_cxcywh_to_xyxy(
            (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        )
        
        # Get classification logits
        all_pred_logits = self.dot_prod_scoring(
            decoder_hidden_states=decoder_output.intermediate_hidden_states,
            text_features=encoder_output.text_features,
            text_mask=prompt_mask,
        ).squeeze(-1)
        
        # Get final layer outputs
        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_output.intermediate_hidden_states[-1]
        presence_logits = decoder_output.presence_logits[-1]
        
        # Get mask predictions
        mask_output = self.segmentation_head(
            decoder_queries=decoder_hidden_states,
            backbone_features=backbone_fpn,
            encoder_hidden_states=encoder_output.last_hidden_state,
            prompt_features=prompt_features,
            prompt_mask=prompt_mask,
        )
        
        return mask_output.pred_masks, pred_boxes, pred_logits, presence_logits

    @staticmethod
    def _inverse_sigmoid(x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    @staticmethod
    def _box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack(
            [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=-1
        )


def export_vision_encoder(model: Sam3Image, output_dir: Path, device: str = "cuda"):
    """Export vision encoder to ONNX."""
    print("Exporting Vision Encoder...")
    wrapper = VisionEncoderWrapper(model, device=device).to(device).eval()

    torch.onnx.export(
        wrapper,
        (torch.randn(1, 3, 1008, 1008, device=device),),
        str(output_dir / "vision-encoder.onnx"),
        input_names=["images"],
        output_names=["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
            "fpn_feat_0": {0: "batch"},
            "fpn_feat_1": {0: "batch"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'vision-encoder.onnx'}")


def export_text_encoder(model: Sam3Image, output_dir: Path, device: str = "cuda"):
    """Export text encoder to ONNX.
    
    Note: SAM3's text encoder takes raw strings, but for ONNX we export
    a version that takes pre-tokenized input_ids and attention_mask.
    Tokenization must be done outside the ONNX model.
    """
    print("Exporting Text Encoder...")
    print("  Note: Text encoder export requires pre-tokenized inputs.")
    print("  You'll need to use SAM3's tokenizer separately before inference.")
    
    wrapper = TextEncoderONNXWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randint(0, 49408, (1, 32), device=device),
            torch.ones(1, 32, dtype=torch.long, device=device),
        ),
        str(output_dir / "text-encoder.onnx"),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_features", "text_mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "text_features": {0: "batch"},
            "text_mask": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'text-encoder.onnx'}")


def export_geometry_encoder(model: Sam3Image, output_dir: Path, device: str = "cuda"):
    """Export geometry encoder to ONNX."""
    print("Exporting Geometry Encoder...")
    wrapper = GeometryEncoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.rand(1, 5, 4, device=device),  # boxes in cxcywh format
            torch.ones(1, 5, dtype=torch.long, device=device),  # box labels
            torch.randn(1, 256, 72, 72, device=device),  # fpn_feat_2
            torch.randn(1, 256, 72, 72, device=device),  # fpn_pos_2
        ),
        str(output_dir / "geometry-encoder.onnx"),
        input_names=["input_boxes", "input_boxes_labels", "fpn_feat_2", "fpn_pos_2"],
        output_names=["geometry_features", "geometry_mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input_boxes": {0: "batch", 1: "num_boxes"},
            "input_boxes_labels": {0: "batch", 1: "num_boxes"},
            "fpn_feat_2": {0: "batch"},
            "fpn_pos_2": {0: "batch"},
            "geometry_features": {0: "batch", 1: "num_prompts"},
            "geometry_mask": {0: "batch", 1: "num_prompts"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'geometry-encoder.onnx'}")


def export_decoder(model: Sam3Image, output_dir: Path, device: str = "cuda"):
    """Export decoder to ONNX."""
    print("Exporting Decoder...")
    wrapper = DecoderWrapper(model).to(device).eval()

    torch.onnx.export(
        wrapper,
        (
            torch.randn(1, 256, 288, 288, device=device),  # fpn_feat_0
            torch.randn(1, 256, 144, 144, device=device),  # fpn_feat_1
            torch.randn(1, 256, 72, 72, device=device),    # fpn_feat_2
            torch.randn(1, 256, 72, 72, device=device),    # fpn_pos_2
            torch.randn(1, 32, 256, device=device),        # prompt_features
            torch.ones(1, 32, dtype=torch.bool, device=device),  # prompt_mask
        ),
        str(output_dir / "decoder.onnx"),
        input_names=[
            "fpn_feat_0",
            "fpn_feat_1",
            "fpn_feat_2",
            "fpn_pos_2",
            "prompt_features",
            "prompt_mask",
        ],
        output_names=["pred_masks", "pred_boxes", "pred_logits", "presence_logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            **{f"fpn_feat_{i}": {0: "batch"} for i in range(3)},
            "fpn_pos_2": {0: "batch"},
            "prompt_features": {0: "batch", 1: "prompt_len"},
            "prompt_mask": {0: "batch", 1: "prompt_len"},
            "pred_masks": {0: "batch"},
            "pred_boxes": {0: "batch"},
            "pred_logits": {0: "batch"},
            "presence_logits": {0: "batch"},
        },
    )
    print(f"  ✓ Saved: {output_dir / 'decoder.onnx'}")


def load_model(checkpoint_path: str = None, device: str = "cpu"):
    """Load SAM3 model, optionally with fine-tuned checkpoint.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint (optional)
        device: Device to load model on
        
    Returns:
        Sam3Image model
    """
    print("Loading SAM3 model...")
    
    # Build the SAM3 image model (downloads from HuggingFace if needed)
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=None if checkpoint_path else None,  # Don't load default if we have custom
        load_from_HF=checkpoint_path is None,  # Only load from HF if no custom checkpoint
    )
    
    if checkpoint_path:
        print(f"Loading fine-tuned weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove 'detector.' prefix if present (from training)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("detector."):
                new_key = key[len("detector."):]
                cleaned_state_dict[new_key] = value
            elif key.startswith("module."):
                new_key = key[len("module."):]
                cleaned_state_dict[new_key] = value
            else:
                cleaned_state_dict[key] = value
        
        # Filter out tracker and sam2_convs keys (video-related)
        filtered_state_dict = {
            k: v for k, v in cleaned_state_dict.items()
            if not k.startswith("tracker") and "sam2_convs" not in k
        }
        
        # Load with strict=False to ignore mismatched keys
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys")
    
    print("  ✓ Model loaded successfully")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM3 model to ONNX format with dynamic batch support"
    )
    parser.add_argument(
        "--module", type=str, choices=["vision", "text", "geometry", "decoder"],
        help="Specific module to export"
    )
    parser.add_argument("--all", action="store_true", help="Export all modules")
    parser.add_argument(
        "--checkpoint", type=str,
        help="Path to fine-tuned checkpoint (e.g., outputs/test_checkpoint.pt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="onnx-models", 
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu/cuda)"
    )
    args = parser.parse_args()

    if not args.module and not args.all:
        parser.error("Please specify --module or --all")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(checkpoint_path=args.checkpoint, device=args.device)

    modules = ["vision", "text", "geometry", "decoder"] if args.all else [args.module]

    with torch.no_grad():
        for m in modules:
            try:
                export_func = {
                    "vision": export_vision_encoder,
                    "text": export_text_encoder,
                    "geometry": export_geometry_encoder,
                    "decoder": export_decoder,
                }[m]
                export_func(model, output_dir, args.device)
            except Exception as e:
                print(f"  ✗ Failed to export {m}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n✓ Export complete! Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
