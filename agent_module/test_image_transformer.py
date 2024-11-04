import json
from tkinter.tix import IMAGE
import requests
import torch
from transformers import pipeline
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from huggingface_hub import login
import supervision as sv
from api import ACCESS_TOKEN
from transformers import DetrImageProcessor
from transformers import (
    DetrForObjectDetection,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
from targets_classifier import TargetsDataset


def annotate(image, annotations, classes):
    labels = [classes[class_id] for class_id in annotations.class_id]
    print(annotations)
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(
        annotated_image, annotations, labels=labels
    )
    return annotated_image


# from agent_module.targets_classifier import TargetsDataset

login(token=ACCESS_TOKEN)

transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["category_id"], clip=True, min_area=1
    ),
)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
checkpoint = "rtdetr_50_finetune_diff_lr\\transformer\\checkpoint-1000"
processor = DetrImageProcessor.from_pretrained(checkpoint, do_resize=True)
# train_dataset = TargetsDataset(
#     "cocodataset\\result.json",
#     "cocodataset\\",
#     processor=image_processor,
#     transform=transform,
#     # device=device,
# )

ds_test = sv.DetectionDataset.from_coco(
    images_directory_path=f"cocodataset/",
    annotations_path=f"cocodataset/test_new.json",
)


id2label = {id: label for id, label in enumerate(ds_test.classes)}
label2id = {label: id for id, label in enumerate(ds_test.classes)}
model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    # num_labels=48,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
).to(device)


IMAGE_COUNT = 5
start: int = int(
    torch.randint(low=0, high=len(ds_test) - IMAGE_COUNT, size=(1,)).item()
)
for i in range(start, start + IMAGE_COUNT):
    path, source_image, annotations = ds_test[i]

    image = Image.open(path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image.size
    results = processor.post_process_object_detection(
        outputs, target_sizes=[(h, w)], threshold=0.1
    )

    detections = sv.Detections.from_transformers(results[0]).with_nms(threshold=0.1)

    annotated_images = [
        annotate(source_image, annotations, ds_test.classes),
        annotate(source_image, detections, ds_test.classes),
    ]
    grid = sv.create_tiles(
        annotated_images,
        titles=["ground truth", "prediction"],
        titles_scale=1,
        single_tile_size=(800, 800),
        tile_padding_color=sv.Color.WHITE,
        tile_margin_color=sv.Color.WHITE,
    )
    sv.plot_image(grid, size=(6, 6))
