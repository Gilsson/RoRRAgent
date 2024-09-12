"""
Image classification for Risk of Rain monsters
"""

import os
import time

import torch
from tqdm import tqdm
import classifier
from ror_api import RorAPI
from targets_classifier import (
    TargetsDataset,
    collate_fn,
    get_model_instance_segmentation,
    get_transform,
)

import torchvision


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

if __name__ == "__main__":
    print("empty")

    # classifier.get_icons("C:\\Users\\anton\\OneDrive\\Pictures\\Test")
    # net = classifier.CooldownCNN()
    # classifier.train_net(net)
    # net.load_state_dict(torch.load("model.pth"))

    # classifier.train_net(net)
    # api = RorAPI(net)
    # for i in range(100):
    # print(api.test_icons())
    # time.sleep(5)
    # api.make_random_screens(100)
    # classifier.crop_images("C:\\Users\\anton\\OneDrive\\Pictures\\Cooldowns")
    # classifier.crop_images(".\\marked")

    # classifier.get_cooldown()
    # while True:
    #     ror_api.jump()
    # if not ror_api.check_if_started():
    #     ror_api.start_game()
    #     time.sleep(30)
    # ror_api.start_game()
    # ror_api.launch_game()
    # ror_api.primary_skill(0.5)
    # ror_api.secondary_skill(0.5)
    # ror_api.utility_skill(0.5)
    # ror_api.ult_skill(0.5)
