"""
Api for calling risk of rain actions
"""

import asyncio
import time
import subprocess
import keyboard
import psutil
import torch
from PIL import ImageGrab, Image

# from ror_inject import RoRInject

RISK_OF_RAIN_PATH = "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Risk of Rain Returns\\Risk of Rain Returns.exe"
GAME_NAME = "Risk of Rain Re"

Z = keyboard.key_to_scan_codes("z")[0]
X = keyboard.key_to_scan_codes("x")[0]
C = keyboard.key_to_scan_codes("c")[0]
V = keyboard.key_to_scan_codes("v")[0]
Z = keyboard.key_to_scan_codes("z")[0]
B = keyboard.key_to_scan_codes("b")[0]
N = keyboard.key_to_scan_codes("n")[0]
M = keyboard.key_to_scan_codes("m")[0]
UP = keyboard.key_to_scan_codes("up")[0]
DOWN = keyboard.key_to_scan_codes("down")[0]
RIGHT = keyboard.key_to_scan_codes("right")[0]
LEFT = keyboard.key_to_scan_codes("left")[0]
SPACE = keyboard.key_to_scan_codes("space")[0]


class Screenshoter:
    def __init__(self):
        self.current_idx = 0
        self.__get_latest_screenshot()

    def __get_latest_screenshot(self):
        import os

        files = os.listdir(".\\screens\\")
        # print(files)
        if files:
            files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
            number = int(files[-1].split(".")[0])
            self.current_idx = number + 1

    def make_screen(self):

        from PIL import ImageGrab

        screenshot = ImageGrab.grab()

        # Save the screenshot
        screenshot.save(f".\\screens\\{self.current_idx}.png")
        self.current_idx += 1

    def get_icons_state(self) -> list[Image.Image]:

        screenshot = ImageGrab.grab()

        icon_regions = [
            (901, 999, 926, 1024),
            (931, 999, 956, 1024),
            (964, 999, 989, 1024),
            (994, 999, 1019, 1024),
        ]

        # Crop icons in a single operation
        icons = [screenshot.crop(region) for region in icon_regions]

        return icons


class RorAPI:

    def __init__(self, icons_net=None):
        import classifier

        self.screenshoter = Screenshoter()

        # self.icons_net = icons_net
        self.icons_net = classifier.CooldownCNN()
        self.icons_net.load_state_dict(torch.load("models/icons_model.pth"))

    def get_cooldowns(self):
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        icons = self.screenshoter.get_icons_state()
        # Apply transformations to all icons in a single batch
        icons_tensor = torch.stack([transform(icon) for icon in icons])

        # Forward pass through the network
        with torch.no_grad():  # Disable gradient computation
            logits = self.icons_net(icons_tensor)

        preds = torch.sigmoid(logits) > 0.5
        preds = torch.tensor(preds, dtype=torch.uint8)
        return preds.tolist()

    def make_screens(self, duration, screen_nums):
        for _ in range(screen_nums):
            self.screenshoter.make_screen()
            time.sleep(duration)

    def make_random_screens(self, screen_nums):
        for _ in range(screen_nums):
            RorAPI.primary_skill(2)
            self.secondary_skill(1)
            self.utility_skill(1)
            self.ult_skill(1)
            self.move_to_left(1)
            self.move_to_right(1)

            self.screenshoter.make_screen()
            time.sleep(3)

    def check_if_started(self):
        """Check if game already started"""
        for process in psutil.process_iter(["pid", "name"]):
            if process.info["name"] == GAME_NAME:
                return True
        return False

    @staticmethod
    def press_and_release_key(key, delta: float):
        """Press a key and release after a delay"""
        keyboard.press(key)
        time.sleep(delta)
        keyboard.release(key)

    @staticmethod
    def press_multiple_keys(keys, delta: float):
        """Press multiple keys and release after a delay"""
        for key in keys:
            keyboard.press(key)
        time.sleep(delta)
        for key in keys:
            keyboard.release(key)

    @staticmethod
    def start_game():
        """Start the game"""
        print(RISK_OF_RAIN_PATH)
        try:
            subprocess.call(RISK_OF_RAIN_PATH, shell=True)
        except FileNotFoundError:
            print(f"The executable '{RISK_OF_RAIN_PATH}' was not found.")

    @staticmethod
    def launch_game():
        """Launch the game"""
        RorAPI.press_and_release_key(DOWN, 0.5)
        RorAPI.press_and_release_key(UP, 0.5)
        RorAPI.press_and_release_key(Z, 0.5)
        time.sleep(1)
        RorAPI.press_and_release_key(DOWN, 2.0)
        RorAPI.press_and_release_key(Z, 0.5)
        time.sleep(4)

    @staticmethod
    def move_to_right(delta: float):
        """Move the player to the right"""
        RorAPI.press_and_release_key(RIGHT, delta)

    @staticmethod
    def move_to_left(delta: float):
        """Move the player to the left"""
        RorAPI.press_and_release_key(LEFT, delta)

    @staticmethod
    def move_to_down(delta: float):
        """Move the player downward"""
        RorAPI.press_and_release_key(DOWN, delta)

    @staticmethod
    def move_to_up(delta: float):
        """Move the player upward"""
        RorAPI.press_and_release_key(UP, delta)

    @staticmethod
    def primary_skill(delta: float):
        """Use the primary skill"""
        RorAPI.press_and_release_key(Z, delta)

    @staticmethod
    def secondary_skill(delta: float):
        """Use the secondary skill"""
        RorAPI.press_and_release_key(X, delta)

    @staticmethod
    def utility_skill(delta: float):
        """Use the utility skill"""
        RorAPI.press_and_release_key(C, delta)

    @staticmethod
    def ult_skill(delta: float):
        """Use the ultimate skill"""
        RorAPI.press_and_release_key(V, delta)

    @staticmethod
    def use_equipment(delta: float):
        """Use the equipment"""
        RorAPI.press_and_release_key(N, delta)

    @staticmethod
    def swap_equipment(delta: float):
        """Swap equipment"""
        RorAPI.press_and_release_key(M, delta)

    @staticmethod
    def jump(delta: float):
        """Make the player jump"""
        RorAPI.press_and_release_key(SPACE, delta)

    @staticmethod
    def use_item(delta: float):
        """Use an item"""
        RorAPI.press_and_release_key(B, delta)

    @staticmethod
    def restart_game():
        """Restart the game"""
        for _ in range(5):
            RorAPI.press_and_release_key(Z, 0.5)
