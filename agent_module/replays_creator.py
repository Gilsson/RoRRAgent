from datetime import datetime
import sys
import keyboard
import os
import time
from time import time as now
import gymnasium as gym

from agent_module.env import RiskOfRainEnv


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "yolov5"))


class ReplayCreator:
    def __init__(self, env):
        keyboard.add_hotkey("ctrl+5", self.start_record)
        keyboard.add_hotkey("ctrl+0", self.stop_record)
        self.recording = False
        self.env: RiskOfRainEnv = env
        self.duration = 0
        self.start_time = 0
        os.chdir("replays")
        self.replay_list = []
        self.state = None

    def make_env_step(self, e: keyboard.KeyboardEvent, duration: float = 0):
        next_state, reward, done, _ = self.env.step((e.scan_code, duration))
        self.replay_list.append(
            {
                "time": e.time,
                "action": e.scan_code,
                "duration": duration,
                "state": self.state,
                "next_state": next_state,
                "reward": reward,
                "done": done,
            }
        )
        self.state = next_state

    def end_count_duration(self, e: keyboard.KeyboardEvent):
        self.duration = now() - self.start_time
        self.make_env_step(e, self.duration)

    def count_duration(self, e):
        self.start_time = time.time()
        keyboard.on_release(lambda: self.end_count_duration(e))

    def start_record(self):
        if not self.recording:
            self.recording = True
            self.state, _ = self.env.reset()
            while self.recording:
                keyboard.on_press(lambda e: self.count_duration(e))

    def stop_record(self):
        if self.recording:
            print("Recording stopped")
            self.recording = False
            list_dir = os.listdir(".")
            print(list_dir)
            largest_num = 0
            for f in list_dir:
                num_replay = int(f.split(".")[0].split("_")[1])
                if num_replay > largest_num:
                    largest_num = num_replay
            fd = os.open(f"replay_{largest_num+1}.txt", os.O_RDWR | os.O_CREAT)
            for event in self.replay_list:
                os.write(
                    fd,
                    (
                        f"[{datetime.fromtimestamp(event['time'])}] [{event['action']}] [{event['duration']}] [{event['state']}] [{event['next_state']}] [{event['next_state']}] [{event['reward']}] [{event['done']}] \n"
                    ).encode(),
                )
            os.close(fd)


gym.register(
    id="RiskOfRain-v0",
    entry_point="env:RiskOfRainEnv",  # Adjust the entry point if defined in another file/module
)
# print(gym.spec("RiskOfRain-v0"))
# print(gym.envs.registry.keys())
env = gym.make("RiskOfRain")
replay_creator = ReplayCreator(env)
keyboard.wait()
