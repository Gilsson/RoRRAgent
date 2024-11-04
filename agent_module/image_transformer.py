import os
import torch
import requests

import numpy as np
import supervision as sv
import albumentations as A

from PIL import Image
from pprint import pprint
from dataclasses import dataclass, replace
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchsummary
import transformers
from transformers.integrations import is_tensorboard_available, TensorBoardCallback

import tensorboard

torch.backends.cuda.matmul.allow_tf32 = True


class PyTorchDetectionDataset(Dataset):
    def __init__(
        self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None
    ):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    @staticmethod
    def annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, image, annotations = self.dataset[idx]

        # Convert image to RGB numpy array
        image = image[:, :, ::-1]
        boxes = annotations.xyxy
        categories = annotations.class_id
        height, width, _ = image.shape
        num_objs = len(boxes)
        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(
                [
                    [
                        min(box[0].item(), width - 2),
                        min(box[1].item(), height - 2),
                        min(box[2].item(), width),
                        min(box[3].item(), height),
                    ]
                    for box in boxes
                ]
            )
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category=categories)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            categories = transformed["category"]

        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=categories, boxes=boxes
        )
        result = self.processor(
            images=image, annotations=formatted_annotations, return_tensors="pt"
        )

        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result


def annotate(image, annotations, classes):
    labels = [classes[class_id] for class_id in annotations.class_id]

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(
        annotated_image, annotations, labels=labels
    )
    return annotated_image


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:

    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(
                np.array(
                    [
                        (
                            x["size"]
                            if len(x["size"] == 2)
                            else x["size"].append(x["size"])
                        )
                        for x in batch
                    ]
                )
            )
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, image_size_batch):
                size = size.tolist()
                if len(size) != 2:
                    size.append(size[0])
                width, height = tuple(size)

                boxes = target["boxes"]
                boxes = sv.xcycwh_to_xyxy(boxes)
                boxes = boxes * np.array([width, height, width, height])
                boxes = torch.tensor(boxes)
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(
                logits=torch.tensor(batch_logits),
                pred_boxes=torch.tensor(batch_boxes),
            )
            for size in target_sizes:
                if len(size) != 2:
                    target_sizes = torch.tensor([size[0].item(), size[0].item()])
            # print(target_sizes)
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):

        predictions, targets = (
            evaluation_results.predictions,
            evaluation_results.label_ids,
        )

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(
            classes, map_per_class, mar_100_per_class
        ):
            class_name = (
                id2label[class_id.item()] if id2label is not None else class_id.item()
            )
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


if __name__ == "__main__":
    CHECKPOINT = "PekingU/rtdetr_r50vd_coco_o365"
    # CHECKPOINT = "rtdetr_101_finetune/backbone/checkpoint-1200"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=f"cocodataset/",
        annotations_path=f"cocodataset/train_new.json",
    )
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=f"cocodataset/",
        annotations_path=f"cocodataset/test_new.json",
    )

    GRID_SIZE = 5

    IMAGE_SIZE = 1280

    processor = AutoImageProcessor.from_pretrained(
        CHECKPOINT,
        # do_resize=True,
        # size={"width": 800, "height": 800},
        do_normalize=True,
    )
    print(processor)

    train_augmentation_and_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(
                max_holes=10,
                max_height=40,
                max_width=40,
                min_holes=1,
                min_height=10,
                min_width=10,
                fill_value=0,
                p=1,
            ),
            A.ColorJitter(p=1.0),
            A.RandomScale(scale_limit=(0.5, 1.5)),
            A.BBoxSafeRandomCrop(height=512, width=512),
            A.Rotate(p=0.5, limit=(-10, 10)),
            A.HueSaturationValue(p=0.5),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
            A.Resize(640, 640),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["category"], min_area=25
        ),
    )

    valid_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["category"], min_area=1
        ),
    )

    IMAGE_COUNT = 0

    for i in range(IMAGE_COUNT):
        _, image, annotations = ds_train[i]

        output = train_augmentation_and_transform(
            image=image, bboxes=annotations.xyxy, category=annotations.class_id
        )

        augmented_image = output["image"]
        augmented_annotations = replace(
            annotations,
            xyxy=np.array(output["bboxes"]),
            class_id=np.array(output["category"]),
        )

        annotated_images = [
            annotate(image, annotations, ds_train.classes),
            annotate(augmented_image, augmented_annotations, ds_train.classes),
        ]
        grid = sv.create_tiles(
            annotated_images,
            titles=["original", "augmented"],
            titles_scale=0.5,
            single_tile_size=(640, 640),
            tile_padding_color=sv.Color.WHITE,
            tile_margin_color=sv.Color.WHITE,
        )
        sv.plot_image(grid, size=(6, 6))
    # exit(0)
    pytorch_dataset_train = PyTorchDetectionDataset(
        ds_train, processor, transform=train_augmentation_and_transform
    )
    pytorch_dataset_valid = PyTorchDetectionDataset(
        ds_test, processor, transform=valid_transform
    )

    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for id, label in enumerate(ds_train.classes)}

    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=processor, threshold=0.01, id2label=id2label
    )

    model = AutoModelForObjectDetection.from_pretrained(
        CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    output_path = "rtdetr_50_finetune_diff_lr"
    # for name, param in model.named_parameters():
    #     print(name)
    #     if name.startswith("model.backbone"):
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    # exit(0)
    # backbone_path = output_path + "/backbone"
    # training_args = TrainingArguments(
    #     output_dir=backbone_path,
    #     num_train_epochs=10,
    #     max_grad_norm=0.1,
    #     learning_rate=1e-4,
    #     warmup_steps=0,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=1,
    #     dataloader_num_workers=4,
    #     metric_for_best_model="eval_map",
    #     greater_is_better=True,
    #     load_best_model_at_end=True,
    #     eval_strategy="steps",
    #     save_strategy="steps",
    #     eval_steps=400,
    #     save_steps=400,
    #     save_total_limit=4,
    #     remove_unused_columns=False,
    #     eval_do_concat_batches=False,
    #     push_to_hub=True,
    #     report_to=["tensorboard"],
    #     logging_strategy="steps",
    #     logging_dir=f"{backbone_path}/runs",
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=pytorch_dataset_train,
    #     eval_dataset=pytorch_dataset_valid,
    #     processing_class=processor,
    #     data_collator=collate_fn,
    #     compute_metrics=eval_compute_metrics_fn,
    # )

    # trainer.train()

    # for name, param in model.named_parameters():
    #     param.requires_grad = True

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if n.startswith("model.backbone")
            ],
            "lr": 5e-6,
            "weight_decay": 0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not n.startswith("model.backbone")
            ],
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=5e-5,
    )
    transformer_path = output_path + "/transformer"
    training_args = TrainingArguments(
        output_dir=transformer_path,
        num_train_epochs=50,
        max_grad_norm=0.1,
        warmup_steps=500,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        dataloader_num_workers=4,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="steps",
        save_strategy="steps",
        # eval_steps=400,
        # save_steps=400,
        save_total_limit=4,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=True,
        report_to=["tensorboard"],
        logging_strategy="steps",
        logging_dir=f"{transformer_path}/runs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        optimizers=(optimizer, None),
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    # try:
    # trainer.train(resume_from_checkpoint=True)
    # except ValueError:
    trainer.train()
    targets = []
    predictions = []

    for i in range(len(ds_test)):
        path, sourece_image, annotations = ds_test[i]

        image = Image.open(path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        w, h = image.size
        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.3
        )

        detections = sv.Detections.from_transformers(results[0])

        targets.append(annotations)
        predictions.append(detections)

    mean_average_precision = sv.MeanAveragePrecision.from_detections(
        predictions=predictions,
        targets=targets,
    )

    print(f"map50_95: {mean_average_precision.map50_95:.2f}")
    print(f"map50: {mean_average_precision.map50:.2f}")
    print(f"map75: {mean_average_precision.map75:.2f}")

    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions, targets=targets, classes=ds_test.classes
    )

    confusion_matrix.plot("confusion_matrix.png")
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    IMAGE_COUNT = 5

    for i in range(IMAGE_COUNT):
        path, sourece_image, annotations = ds_test[i]

        image = Image.open(path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        w, h = image.size
        results = processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=0.1
        )

        detections = sv.Detections.from_transformers(results[0]).with_nms(threshold=0.1)

        annotated_images = [
            annotate(sourece_image, annotations, ds_train.classes),
            annotate(sourece_image, detections, ds_train.classes),
        ]
        grid = sv.create_tiles(
            annotated_images,
            titles=["ground truth", "prediction"],
            titles_scale=0.5,
            single_tile_size=(400, 400),
            tile_padding_color=sv.Color.WHITE,
            tile_margin_color=sv.Color.WHITE,
        )
        sv.plot_image(grid, size=(6, 6))
