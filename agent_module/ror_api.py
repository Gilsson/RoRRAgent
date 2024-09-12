"""
Api for calling risk of rain actions
"""

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
        print(files)
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
            self.primary_skill(2)
            self.secondary_skill(1)
            self.primary_skill(2)

            self.screenshoter.make_screen()
            self.utility_skill(1)
            self.primary_skill(2)
            self.ult_skill(1)
            self.primary_skill(2)

            self.screenshoter.make_screen()
            self.move_to_left(1)
            self.primary_skill(2)
            self.move_to_right(1)
            self.primary_skill(2)
            self.screenshoter.make_screen()
            time.sleep(3)

    def check_if_started(self):
        """Check if game already started

        Returns:
            bool: If process loaded -- True, else -- False
        """
        for process in psutil.process_iter(["pid", "name"]):
            if process.info["name"] == GAME_NAME:
                return True
        return False

    def press_and_release_key(self, key, delta: float):
        """
        function to press a key and release after delta time
        """
        keyboard.press(key)
        time.sleep(delta)
        keyboard.release(key)

    def start_game(self):
        """
        function to start the game
        """
        print(RISK_OF_RAIN_PATH)
        try:
            # subprocess.call(
            #    "sudo -u gilsson steam steam://rungameid/1337520", shell=True)
            subprocess.call(RISK_OF_RAIN_PATH, shell=True)
        except FileNotFoundError:
            print(f"The executable '{RISK_OF_RAIN_PATH}' was not found.")

    def launch_game(self):
        """
        function to launch the game
        """
        self.press_and_release_key(DOWN, 0.5)
        self.press_and_release_key(UP, 0.5)
        self.press_and_release_key(Z, 0.5)
        time.sleep(1)
        self.press_and_release_key(DOWN, 2.0)
        self.press_and_release_key(Z, 0.5)
        time.sleep(4)

    def move_to_right(self, delta: float):
        """Move the player to the right

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(RIGHT, delta)

    def move_to_left(self, delta: float):
        """Move the player to the left

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(LEFT, delta)

    def move_to_down(self, delta: float):
        """Move the player to the down

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(DOWN, delta)

    def move_to_up(self, delta: float):
        """Move the player to the up

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(UP, delta)

    def primary_skill(self, delta: float):
        """Press button to call primary skill

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(Z, delta)

    def secondary_skill(self, delta: float):
        """Press button to call secondary skill

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(X, delta)

    def utility_skill(self, delta: float):
        """Press button to call utility

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(C, delta)

    def ult_skill(self, delta: float):
        """Press button to call ultimate

        Args:
            delta (float): time to hold button
        """
        self.press_and_release_key(V, delta)

    def use_equipment(self, delta: float):
        """Press button to use equipment"""
        self.press_and_release_key(N, 0.5)

    def swap_equipment(self, delta: float):
        """Press button to swap equipment"""
        self.press_and_release_key(M, 0.5)

    def jump(self, delta: float):
        """Press button to jump"""
        self.press_and_release_key(SPACE, 0.5)

    def use_item(self, delta: float):
        """Press button to use item"""
        self.press_and_release_key(B, 0.5)

    def restart_game(self):
        """Press button to restart game"""
        self.press_and_release_key(Z, 0.5)
        self.press_and_release_key(Z, 0.5)
        self.press_and_release_key(Z, 0.5)
        self.press_and_release_key(Z, 0.5)
        self.press_and_release_key(Z, 0.5)
