import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
image_dir = Path("cocodataset/images")
annotation_path = Path("cocodataset/result.json")
train_annotation_path = Path("cocodataset/train_new.json")
test_annotation_path = Path("cocodataset/test_new.json")

# Load the COCO annotations
with open(annotation_path, "r") as f:
    coco_data = json.load(f)

# Step 1: Identify used classes
used_category_ids = set()
for annotation in coco_data["annotations"]:
    used_category_ids.add(annotation["category_id"])

# Filter categories to keep only those that are used
used_categories = [
    cat for cat in coco_data["categories"] if cat["id"] in used_category_ids
]

# Create a mapping for remapping category IDs to be contiguous
id_mapping = {cat["id"]: new_id for new_id, cat in enumerate(used_categories)}

# Step 2: Filter annotations based on used categories and remap category IDs
filtered_annotations = []
for ann in coco_data["annotations"]:
    if ann["category_id"] in used_category_ids:
        # Remap category ID to new ID
        remapped_ann = (
            ann.copy()
        )  # Copy original annotation to avoid modifying it directly
        remapped_ann["category_id"] = id_mapping[ann["category_id"]]
        filtered_annotations.append(remapped_ann)

# Update coco_data with filtered annotations and categories
filtered_coco_data = {
    "images": coco_data["images"],
    "annotations": filtered_annotations,
    "categories": [
        {"id": new_id, "name": cat["name"]}
        for new_id, cat in enumerate(used_categories)
    ],  # Remap category IDs here as well
}

# Step 3: Split images into train and test sets
image_ids = [img["id"] for img in filtered_coco_data["images"]]
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
train_data = filter_by_image_ids(filtered_coco_data, train_ids)
test_data = filter_by_image_ids(filtered_coco_data, test_ids)

# Save new train and test JSON files
with open(train_annotation_path, "w") as f:
    json.dump(train_data, f)

with open(test_annotation_path, "w") as f:
    json.dump(test_data, f)

print(f"Training annotations saved to {train_annotation_path}")
print(f"Testing annotations saved to {test_annotation_path}")
