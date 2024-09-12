"""
Import torch for training classification problem
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm


"""
Multipliers for learning scores:
"""
MULTIPLIERS = {"gold": 0.05, "time": 0.1, "stage": 1.0}  # stage - 1


def crop_images(root_dir):
    image_filenames = [
        filename for filename in os.listdir(root_dir) if filename.endswith(".png")
    ]
    for idx in range(len(image_filenames)):
        img_filename = os.path.join(root_dir, image_filenames[idx])
        image = Image.open(img_filename).convert("RGB")

        if image.size != (1920, 1080):
            continue
        cropped_image = image.crop((900, 940, 1024, 1064))
        cropped_image.save(os.path.join(root_dir, "cropped_" + image_filenames[idx]))


def get_icons(root_dir):
    image_filenames = [
        filename
        for filename in os.listdir(root_dir)
        if filename.endswith(".png") and filename.startswith("cropped_")
    ]
    for file in os.listdir(root_dir):
        if "icon" in file:
            os.remove(os.path.join(root_dir, file))
    for idx in range(len(image_filenames)):
        img_filename = os.path.join(root_dir, image_filenames[idx])
        image = Image.open(img_filename).convert("RGB")
        ans = img_filename.split("-")[1].split(".")[0]
        if image.size != (124, 124):
            continue
        cropped_image = image.crop((1, 59, 26, 84))
        cropped_image.save(os.path.join(root_dir, f"icon_1_{idx}-{ans[0]}.png"))
        cropped_image = image.crop((31, 59, 56, 84))
        cropped_image.save(os.path.join(root_dir, f"icon_2_{idx}-{ans[1]}.png"))
        cropped_image = image.crop((64, 59, 89, 84))
        cropped_image.save(os.path.join(root_dir, f"icon_3_{idx}-{ans[2]}.png"))
        cropped_image = image.crop((94, 59, 119, 84))
        cropped_image.save(os.path.join(root_dir, f"icon_4_{idx}-{ans[3]}.png"))


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Get a list of all image filenames in the root directory
        self.image_filenames = [
            filename
            for filename in os.listdir(root_dir)
            if filename.endswith(".png") and filename.startswith("icon")
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.root_dir, self.image_filenames[idx])
        image = Image.open(img_filename).convert("RGB")
        # cropped_image = image.crop((900, 940, 1024, 1064))
        # cropped_image.save("cropped_image.png")
        # Extract the class label from the filename
        digit = self.image_filenames[idx].split("-")[1].split(".")[0]

        # Convert the digits to a list of integers
        digit = torch.tensor(int(digit))
        # print(digit)

        if self.transform:
            image = self.transform(image)

        return image, digit


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout=0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)
        x = self.dropout(x)

        return x


class CooldownCNN(nn.Module):
    def __init__(self) -> None:
        super(CooldownCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer added
            nn.Conv2d(25, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def predict_icons(self, icons):
        res = []
        for image in icons:
            output = self(image)
            output = torch.sigmoid(output) > 0.5
            res.append(output)
        return output


def get_cuda_info() -> bool:
    """Get info if cuda is available

    Returns:
        bool: wheter cuda is available
    """
    return torch.cuda.is_available()


def train_net(net: nn.Module):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    net = net.to(device)
    datasets_folder = "C:\\Users\\anton\\OneDrive\\Pictures\\Cooldowns"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_dataset = CustomDataset(root_dir=datasets_folder, transform=transform)

    test_folder = "C:\\Users\\anton\\OneDrive\\Pictures\\Test"
    criterion = torch.nn.BCEWithLogitsLoss()
    data_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataset = CustomDataset(root_dir=test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    train_losses = []
    train_precisions = []
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        epoch_precision = 0
        correct_predictions = 0
        total_samples = 0
        running_loss = 0
        progress_bar = tqdm(
            data_loader,
            desc=f"Epoch {epoch + 1}/{epochs}, Precision {epoch_precision:.4f}",
        )
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            print(images.shape)
            optimizer.zero_grad()
            labels = labels.unsqueeze(1)
            labels = labels.float()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Compute precision
            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            # print(predicted_labels, labels)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            print(total_samples)
            epoch_precision = correct_predictions / total_samples
            running_loss += loss.item()
            progress_bar.set_description(
                f"Epoch {epoch+1}/{epochs}, Precision {epoch_precision:.4f}"
            )
        epoch_loss = running_loss / len(data_loader)
        epoch_precision = correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_precisions.append(epoch_precision)

        print(
            f"Epoch [{epoch + 1}/{10}], Loss: {epoch_loss}, Precision: {epoch_precision}"
        )
    net.eval()
    epoch_precision = 0
    correct_predictions = 0
    total_samples = 0
    running_loss = 0
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        epoch_precision = correct_predictions / total_samples
        running_loss += loss.item()
    epoch_loss = running_loss / len(data_loader)
    epoch_precision = correct_predictions / total_samples
    train_losses.append(epoch_loss)
    train_precisions.append(epoch_precision)
    # torch.save(net.state_dict(), "model.pth")

    print(f"Evaluate: Loss: {epoch_loss}, Precision: {epoch_precision}")


# def get_cooldown():

# 900;984:1026,1032
# print(result)
