"""
Convert individual SAM JSON annotation files to COCO format.
This script consolidates all individual JSON files in data/train and data/valid
into single COCO-format annotation files (_annotations.coco.json).
"""

import json
import os
import re
from pathlib import Path


def transform_filename_to_match_renamed(original_name):
    """
    Transform the original file name from JSON to match the renamed format.
    
    Original format in JSON: -76-3-_png_jpg.rf.4847dccd1aedf5e8b850164f6d76fc29.jpg
    Renamed format on disk:  -76-3-_png_jpg_rf_4847dccd1aedf5e8b850164f6d76fc29_1.jpg
    
    The rename script:
    1. Replaces all dots except the last one with underscores
    2. Adds _1 suffix before extension if no _N pattern exists
    """
    # Count dots
    dot_count = original_name.count(".")
    
    # Replace all except last dot with underscore
    new_name = original_name.replace(".", "_", dot_count - 1)
    
    # Add _1 suffix if no _N pattern at end
    if not re.search(r"_\d+\.\w+$", new_name):
        new_name = new_name.replace(".", "_1.")
    
    return new_name


def convert_folder_to_coco(folder_path):
    """Convert all JSON files in a folder to COCO format."""
    folder = Path(folder_path)
    
    # Get all JSON files
    json_files = sorted(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder}")
        return
    
    print(f"Found {len(json_files)} JSON files in {folder}")
    
    # Initialize COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}]  # Generic category
    }
    
    image_id = 1
    annotation_id = 1
    
    # Process each JSON file
    print(f"Processing {folder.name}...")
    for i, json_file in enumerate(json_files, 1):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(json_files)} files...")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract image info
            image_info = data.get("image", {})
            
            # Get original file name and transform to match renamed format
            original_file_name = image_info.get("file_name", json_file.stem + ".jpg")
            transformed_file_name = transform_filename_to_match_renamed(original_file_name)
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": transformed_file_name,
                "width": image_info.get("width", 1024),
                "height": image_info.get("height", 1024)
            })
            
            # Add annotations
            annotations = data.get("annotations", [])
            for ann in annotations:
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,  # Default category
                    "bbox": ann.get("bbox", [0, 0, 0, 0]),
                    "area": ann.get("area", 0),
                    "segmentation": ann.get("segmentation", {}),
                    "iscrowd": 0
                }
                
                coco_data["annotations"].append(coco_annotation)
                annotation_id += 1
            
            image_id += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    # Save COCO format file
    output_file = folder / "_annotations.coco.json"
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"✓ Created {output_file}")
    print(f"  - {len(coco_data['images'])} images")
    print(f"  - {len(coco_data['annotations'])} annotations")
    print()

def main():
    """Convert train, valid, and test folders."""
    base_path = Path("data")
    
    folders = ["train", "valid", "test"]
    
    for folder_name in folders:
        folder_path = base_path / folder_name
        if folder_path.exists():
            convert_folder_to_coco(folder_path)
        else:
            print(f"Folder not found: {folder_path}")

if __name__ == "__main__":
    main()
