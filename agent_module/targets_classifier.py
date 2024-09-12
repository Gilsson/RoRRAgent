import json
import os
import random
import cv2
from matplotlib import patches, pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import torch.utils
import torch.distributed as dist
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2

MODEL_PATH = ".\\models\\"


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        num_classes=num_classes
    )

    # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    list_dir = os.listdir(MODEL_PATH)
    list_dir = [
        f
        for f in list_dir
        if os.path.isfile(os.path.join(MODEL_PATH, f))
        and f.endswith(".pth")
        and f.startswith("enemy_model_")
    ]
    if not list_dir:
        model_index = 0

    else:
        model_index = sorted(
            [int(file.split(".")[0].split("_")[2]) for file in list_dir]
        )[-1]
        model_name = os.path.join(MODEL_PATH, f"enemy_model_{model_index}.pth")
        print(model_name)
        model.load_state_dict(torch.load(model_name))

    return model, model_index


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class FasterRCNNLoss(torch.nn.Module):
    def __init__(self):
        super(FasterRCNNLoss, self).__init__()

    def forward(self, predictions, targets):
        # Unpack predictions
        boxes_pred = predictions["boxes"]
        labels_pred = predictions["labels"]

        # Unpack targets
        boxes_true = targets["boxes"]
        labels_true = targets["labels"]

        # Localization loss (Smooth L1 loss)
        localization_loss = F.smooth_l1_loss(boxes_pred, boxes_true)

        # Classification loss (Cross-Entropy loss)
        classification_loss = F.cross_entropy(labels_pred, labels_true)

        # Total loss
        total_loss = localization_loss + classification_loss

        return total_loss


def collate_fn(batch):
    return tuple(zip(*batch))


class TargetsDataset(Dataset):
    def __init__(
        self, annotations_file, image_dir, transform=None, device=torch.device("cpu")
    ):
        super(TargetsDataset, self).__init__()
        self.device = device
        self.annotations_file = annotations_file
        self.image_dir = image_dir
        self.transform = transform

        # Load COCO annotations from file
        with open(annotations_file, "r") as f:
            self.coco_data = json.load(f)

        # Preprocess annotations
        self.images = self.coco_data["images"]
        self.annotations = self.coco_data["annotations"]
        self.categories = self.coco_data["categories"]
        self.category_id_to_label = {
            category["id"]: i for i, category in enumerate(self.categories)
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # device = torch.device("cpu")
        image_info = self.images[idx]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image_data = Image.open(image_path).convert("RGB")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # if self.transform:
        #     image_tensor: torch.Tensor = self.transform(image)

        # image = np.array(image, dtype=float)
        # image = image / 255.0
        # Load annotations for the image
        image_annotations = [
            annotation
            for annotation in self.annotations
            if annotation["image_id"] == image_info["id"]
        ]
        boxes = [annotation["bbox"] for annotation in image_annotations]

        labels = torch.tensor(
            [
                self.category_id_to_label[annotation["category_id"]]
                for annotation in image_annotations
            ]
        ).long()
        width = image_data.width
        height = image_data.height
        num_objs = len(boxes)
        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.tensor(
                [
                    [
                        box[0] / width,
                        box[1] / height,
                        (box[0] + box[2]) / width,
                        (box[1] + box[3]) / height,
                    ]
                    for box in boxes
                ]
            )
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, category_ids=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = transformed["category_ids"]
        # image_tensor, boxes = HorizontalFlipTransform()(image_tensor, boxes)
        # draw = ImageDraw.Draw(image)
        # for box, label in zip(boxes, labels):

        #     xmin, ymin, xmax, ymax = box
        #     draw.rectangle(
        #         [
        #             xmin * width,
        #             ymin * height,
        #             xmax * width,
        #             ymax * height,
        #         ],
        #         outline="red",
        #     )
        #     draw.text(
        #         (xmin * width, ymin * height),
        #         f"{label.item()}",
        #         fill="red",
        #     )

        # # Display or save the image with drawn bounding boxes
        # image.show()
        return image.to(device), {
            "boxes": torch.tensor(boxes, dtype=torch.float32).to(self.device),
            "labels": torch.tensor(labels, dtype=torch.int64).to(self.device),
        }


class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer added
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)


class AnchorGenerator:
    def __init__(
        self, num_anchors=9, scales=[1.0, 2.0, 3.0], aspect_ratios=[0.5, 1.0, 2.0]
    ):
        self.num_anchors = num_anchors
        self.scales = scales
        self.aspect_ratios = aspect_ratios

    def generate_anchors(self, feature_map_size, stride=2):
        anchors = []

        for y in range(feature_map_size[0]):
            for x in range(feature_map_size[1]):
                for scale in self.scales:
                    for aspect_ratio in self.aspect_ratios:
                        anchor_width = scale * aspect_ratio
                        anchor_height = scale / aspect_ratio

                        # Compute anchor box coordinates
                        xc = (x + 0.5) * stride
                        yc = (y + 0.5) * stride
                        xmin = xc - 0.5 * anchor_width
                        ymin = yc - 0.5 * anchor_height
                        xmax = xc + 0.5 * anchor_width
                        ymax = yc + 0.5 * anchor_height

                        anchors.append([xmin, ymin, xmax, ymax])

        return torch.tensor(anchors)


class RPN(nn.Module):
    def __init__(self, num_anchors):
        super(RPN, self).__init__()
        self.conv = BackBone()
        self.cls_layer = nn.Conv2d(
            1024, num_anchors * 2, kernel_size=1
        )  # 2 scores for each anchor (objectness score)
        self.reg_layer = nn.Conv2d(
            1024, num_anchors * 4, kernel_size=1
        )  # 4 regression values for each anchor (bounding box coordinates)
        self.anchor_generator = AnchorGenerator()

    def forward(self, x):
        # Apply convolutional layer
        x = self.conv(x)

        # Predict objectness scores for each anchor
        cls_scores = self.cls_layer(x)

        # Predict bounding box regression values for each anchor
        reg_values = self.reg_layer(x)

        # Reshape predictions to match anchor dimensions
        cls_scores = cls_scores.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 2)
        reg_values = reg_values.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        anchor_boxes = self.anchor_generator.generate_anchors(x.size())

        return cls_scores, reg_values, anchor_boxes


def decode_predictions(cls_scores, reg_values, anchor_boxes):
    """
    Decode RPN predictions to obtain proposed bounding box coordinates.

    Args:
    - cls_scores (Tensor): Objectness scores for each anchor (shape: batch_size x num_anchors x 2).
    - reg_values (Tensor): Bounding box regression values for each anchor (shape: batch_size x num_anchors x 4).
    - anchor_boxes (Tensor): Anchor boxes (shape: num_anchors x 4).

    Returns:
    - proposed_boxes (list of Tensors): List of proposed bounding boxes for each image in the batch.
    """
    batch_size, num_anchors, _ = cls_scores.size()
    print(batch_size, num_anchors)

    proposed_boxes = []

    # Iterate over each image in the batch
    for i in range(batch_size):
        # Get objectness scores and regression values for the current image
        cls_scores_i = cls_scores[i]  # Shape: num_anchors x 2
        reg_values_i = reg_values[i]  # Shape: num_anchors x 4

        # Apply softmax to objectness scores to get probabilities
        objectness_probs = F.softmax(cls_scores_i, dim=1)[
            :, 1
        ]  # Get probability of objectness

        # Decode regression values to obtain proposed bounding box coordinates
        proposed_boxes_i = []
        for j in range(num_anchors):
            # Get the j-th anchor box and its corresponding regression values
            anchor_box = anchor_boxes[j]
            reg_values_j = reg_values_i[j]

            # Decode regression values (offsets) to obtain bounding box coordinates
            decoded_box = decode_box(anchor_box, reg_values_j)

            # Apply objectness score as confidence to the decoded box
            score = objectness_probs[j].item()
            decoded_box_with_score = [score * x for x in decoded_box]

            proposed_boxes_i.append(decoded_box_with_score)

        proposed_boxes.append(proposed_boxes_i)

    return proposed_boxes


def decode_box(anchor_box, reg_values):
    """
    Decode regression values (offsets) to obtain bounding box coordinates.

    Args:
    - anchor_box (Tensor): Anchor box (shape: 4).
    - reg_values (Tensor): Regression values (shape: 4).

    Returns:
    - decoded_box (list): Decoded bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    # Unpack anchor box coordinates
    xa, ya, wa, ha = anchor_box

    # Unpack regression values
    dx, dy, dw, dh = reg_values

    # Compute decoded bounding box coordinates
    xc = xa + wa * dx
    yc = ya + ha * dy
    wc = wa * torch.exp(dw)
    hc = ha * torch.exp(dh)

    # Compute decoded bounding box coordinates
    xmin = xc - 0.5 * wc
    ymin = yc - 0.5 * hc
    xmax = xc + 0.5 * wc
    ymax = yc + 0.5 * hc

    decoded_box = [xmin, ymin, xmax, ymax]
    return decoded_box


def plot_bounding_boxes(image, targets, image_size, predictions=None):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    print(image_size)
    height = image_size[1]
    width = image_size[2]
    print(targets)
    for target in targets:
        for box in target["boxes"]:
            rect = patches.Rectangle(
                (box[0] * width, box[1] * height),
                (box[2] - box[0]) * width,
                (box[3] - box[1]) * height,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)

    if predictions:
        for pred in predictions:
            for box in pred["boxes"]:
                rect = patches.Rectangle(
                    (box[0] * width, box[1] * height),
                    (box[2] - box[0]) * width,
                    (box[3] - box[1]) * height,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)

    plt.show()


class HorizontalFlipTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes):
        if random.random() < self.prob:
            image = T.functional.hflip(image)
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        return image, boxes


def train_net(model: nn.Module, model_index, num_epochs):
    # transform = T.Compose(
    #     [
    #         torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
    #     ]
    # )
    transform = A.Compose(
        [
            A.RandomCrop(width=450, height=450),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.ColorJitter(p=0.2),
            A.Rotate(limit=40, p=0.5),
            A.RandomSizedBBoxSafeCrop(height=512, width=512, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
    )

    train_dataset = TargetsDataset(
        "result.json", ".\\", transform=transform, device=device
    )
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        loss_epoch = 0
        range_tqdm = tqdm(
            train_loader, desc=f"Epoch[{epoch + 1}/{num_epochs}] Loss {loss_epoch}"
        )
        for images, targets in range_tqdm:
            # print(images)
            # print(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            # print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()

            range_tqdm.set_description(
                f"Epoch[{epoch + 1}/{num_epochs}] Loss {loss_value}"
            )
            losses.backward()
            optimizer.step()
            lr_scheduler.step()
            # model.eval()
            # with torch.no_grad():
            #     sample_image = images[0].cpu().numpy().transpose(1, 2, 0)
            #     sample_targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            #     predictions = model([images[0]])
            #     predictions = [{k: v.cpu() for k, v in p.items()} for p in predictions]

            #     plot_bounding_boxes(
            #         sample_image, sample_targets, images[0].size(), predictions
            #     )
            # model.train()

        if (epoch + 1) % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_PATH, f"enemy_model_{model_index + epoch + 1}.pth"),
            )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 48
    model, model_index = get_model_instance_segmentation(num_classes)
    model = model.to(device)
    # train_net(model, model_index, 100)

    model.eval()
    test_image_path = "7.png"
    input_image = cv2.imread(test_image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    # transform = T.Compose(
    #     [
    #         T.ToImage(),
    #         T.ToDtype(torch.float32, scale=True),
    #         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    # transform = (
    #     torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    # )

    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )
    # Apply the transformation to the image
    input_image = transform(image=input_image)["image"].unsqueeze(0).to(device)
    # Perform inference on the input image
    with torch.no_grad():
        prediction = model(input_image)
    print(prediction)

    boxes = prediction[0]["boxes"].cpu().numpy()  # Extract predicted bounding boxes
    labels = prediction[0]["labels"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()  # Extract confidence scores
    image = Image.open(test_image_path).convert("RGB")
    # Draw bounding boxes on original image
    draw = ImageDraw.Draw(image)
    # for box in boxes:
    #     xmin, ymin, xmax, ymax = box
    #     draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    print(labels)
    for box, label, score in zip(boxes, labels, scores):

        xmin, ymin, xmax, ymax = box.astype(int)
        print(box)
        draw.rectangle(
            [
                xmin * image.width,
                ymin * image.height,
                xmax * image.width,
                ymax * image.height,
            ],
            outline="red",
        )
        draw.text(
            (xmin * image.width, ymin * image.height),
            f"{label.item()}",
            fill="red",
        )

    # Display or save the image with drawn bounding boxes
    image.show()
    image.save(".\\test_box_test.png")
