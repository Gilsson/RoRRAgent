import json
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
image_dir = Path("cocodataset/images")
annotation_path = Path("cocodataset/result.json")
train_annotation_path = Path("cocodataset/train.json")
test_annotation_path = Path("cocodataset/test.json")

# Load the COCO annotations
with open(annotation_path, "r") as f:
    coco_data = json.load(f)

# Split images into train and test sets
image_ids = [img["id"] for img in coco_data["images"]]
train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)


# Helper function to filter by image IDs
def filter_by_image_ids(data, image_ids):
    images = [img for img in data["images"] if img["id"] in image_ids]
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids]
    return {
        "images": images,
        "annotations": annotations,
        "categories": data["categories"],
    }


# Create train and test annotations
train_data = filter_by_image_ids(coco_data, train_ids)
test_data = filter_by_image_ids(coco_data, test_ids)

# Save new train and test JSON files
with open(train_annotation_path, "w") as f:
    json.dump(train_data, f)

with open(test_annotation_path, "w") as f:
    json.dump(test_data, f)

print(f"Training annotations saved to {train_annotation_path}")
print(f"Testing annotations saved to {test_annotation_path}")
