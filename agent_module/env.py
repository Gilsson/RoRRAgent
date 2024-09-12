import cv2
import gymnasium as gym
from gymnasium import spaces
import imageio
import numpy as np
from PIL import Image
from sympy import im
import torch
from ror_inject import RoRInject
from ror_api import RorAPI
import sys
import os

from gymnasium.envs.registration import register

from models.common import Detections
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from utils.general import scale_boxes
from PIL import ImageGrab, Image

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov5"))
)

from models.common import DetectMultiBackend  # type: ignore
from utils.general import non_max_suppression  # type: ignore
from utils.augmentations import letterbox  # type: ignore
from models.experimental import attempt_load


def load_model(weights_path, device):
    model = attempt_load(weights_path, map_location=device)
    if device.type != "cpu":
        model.half()
    return model


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Rescale coords (xyxy) from img1_shape to img0_shape
    :param img1_shape: Shape of the input image (used for model inference)
    :param coords: Bounding box coordinates (x1, y1, x2, y2)
    :param img0_shape: Original image shape
    :param ratio_pad: Tuple of (ratio, pad) if provided, otherwise computes from img1_shape and img0_shape
    :return: Rescaled bounding box coordinates
    """
    if ratio_pad is None:  # Calculate from img1_shape and img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # Gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain, pad = ratio_pad

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4].clamp_(
        min=0, max=img0_shape[0] if img0_shape[0] > img0_shape[1] else img0_shape[1]
    )  # avoid out of bounds
    return coords


class RiskOfRainEnv(gym.Env):
    def __init__(self, verbose=False):
        super(RiskOfRainEnv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = DetectMultiBackend(
        #     "models\\best_yolo.pt",
        #     device=self.device,
        #     data="yolov5\\data\\data.yaml",
        # )
        self.model = torch.hub.load(
            "yolov5", "custom", path="models\\best_yolo.pt", source="local"
        ).to(self.device)
        self.verbose = verbose
        self.input_size = (640, 384)
        self.api = RorAPI()
        self.inject = RoRInject()
        self.inject.update()
        self.action_space = spaces.Discrete(12)  # 11 actions defined in RorAPI
        self.observation_space = spaces.Dict(
            {
                "cooldowns": spaces.MultiBinary(4),
                "health": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "money": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "previous_health": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "detections": spaces.Sequence(
                    spaces.Tuple(
                        (
                            spaces.Box(
                                low=0, high=np.inf, shape=(4,), dtype=np.float32
                            ),  # bbox (x1, y1, x2, y2)
                            spaces.Discrete(48),  # class label
                            spaces.Box(
                                low=0, high=1, shape=(1,), dtype=np.float32
                            ),  # confidence
                        )
                    )
                ),
            }
        )

    def reset(self):
        self.api.restart_game()
        # self.api.launch_game()
        state, _ = self.get_observation()
        reward = self.compute_reward(state)
        if self.verbose:
            print(reward)
        return state, {}

    def step(self, action):
        self.perform_action(action)
        state, _ = self.get_observation()
        reward = self.compute_reward(state)
        done = self.is_done(state)
        info = {}
        return state, reward, done, info

    def render(self, mode="human"):
        # Optionally implement rendering if needed
        pass

    def close(self):
        # Optionally implement clean-up
        pass

    def get_observation(self):
        icons = self.api.get_cooldowns()

        screenshot = ImageGrab.grab()
        # screenshot = Image.open(".\\datasets\\yolovdata\\images\\val\\0a6bfeb3-665.png")
        # screenshot.show()
        # detections = self.run_yolo_inference(screenshot)
        # self.detect_and_save("images_temp\\test\\test.png")
        prev_health = self.inject.health
        self.inject.update()
        detections = self.detect_and_save("images_temp\\test\\test.png")
        if self.verbose:
            print(detections)
        # Combine icons into one image for simplicity
        state = {
            "cooldowns": np.hstack([np.array(icon) for icon in icons]),
            "health": self.inject.health,
            "money": self.inject.money,
            "time": self.inject.time,
            "previous_health": prev_health,
            "detections": detections,
        }
        return state, {}

    def perform_action(self, action):
        action_index, duration = action
        actions = [
            self.api.move_to_right,
            self.api.move_to_left,
            self.api.move_to_down,
            self.api.move_to_up,
            self.api.primary_skill,
            self.api.secondary_skill,
            self.api.utility_skill,
            self.api.ult_skill,
            self.api.use_equipment,
            self.api.swap_equipment,
            self.api.use_item,
            self.api.jump,
        ]
        action2text = {
            0: "Move right",
            1: "Move left",
            2: "Move down",
            3: "Move up",
            4: "Primary skill",
            5: "Secondary skill",
            6: "Utility skill",
            7: "Ultimate skill",
            8: "Use equipment",
            9: "Swap equipment",
            10: "Use item",
            11: "Jump",
        }
        # Perform the action
        if action_index in range(len(actions)):
            actions[action_index](abs(duration))
            print(
                f"Action {action2text[action_index]} performed for {duration} seconds."
            )
            # pass

    def compute_reward(self, state):
        reward = 0.0
        # Define reward function based on the state
        reward -= state["money"] * 0.01
        reward += state["time"] * 0.02
        health_diff = state["health"] - state["previous_health"]
        if health_diff > 0:
            reward += health_diff * 0.01  # Positive reward for gaining health
        else:
            reward += health_diff * 0.15  # Larger penalty for losing health
        print(len(state["detections"]))
        reward -= len(state["detections"]) * 0.2
        # if self.verbose:
        reward = np.clip(reward, -10, 10)
        print(f"reward {reward}")
        return reward

    def capture_screen(self):
        # Capture screen using PIL's ImageGrab
        screen = ImageGrab.grab()
        screen_np = np.array(screen)
        return screen_np

    def resize_image(self, img, size):
        return cv2.resize(img, size)

    def run_inference(self, img):
        # Convert the image to a format suitable for the model
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Run inference
        results = self.model(img_pil)
        return results

    def get_mobs_bbox(self, pred, scaled_img, screenshot):
        from torchvision.ops.boxes import nms

        if self.verbose:
            print(pred)
        boxes = pred[:, :4]
        confidences = pred[:, 4]

        # Perform NMS
        keep = nms(boxes, confidences, iou_threshold=0.45)
        pred = pred[keep]
        bboxes = pred[:, :4].cpu().numpy()
        confidences = pred[:, 4].cpu().numpy()
        class_indices = pred[:, 5].cpu().numpy().astype(int)

        # Calculate the scaling factors
        height_scale = screenshot.shape[0] / scaled_img.shape[0]
        width_scale = screenshot.shape[1] / scaled_img.shape[1]
        detections = []
        for bbox, cls, conf in zip(bboxes, class_indices, confidences):
            # Scale bounding boxes back to the original image size
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * width_scale)
            y1 = int(y1 * height_scale)
            x2 = int(x2 * width_scale)
            y2 = int(y2 * height_scale)
            xyxy = [x1, y1, x2, y2]
            bbox = torch.tensor(xyxy).cpu().numpy()
            detections.append(
                {"bbox": bbox, "class": int(cls), "confidence": float(conf)}
            )

        return detections

    def draw_detections(self, original_img, resized_img, results):
        # Extract predictions
        pred = results.pred[0]

        # Filter by confidence (optional)
        pred = pred[pred[:, 4] > 0.25]  # Confidence threshold

        bboxes = pred[:, :4].cpu().numpy()
        confidences = pred[:, 4].cpu().numpy()
        class_indices = pred[:, 5].cpu().numpy().astype(int)
        class_names = [results.names[i] for i in class_indices]

        # Calculate the scaling factors
        height_scale = original_img.shape[0] / resized_img.shape[0]
        width_scale = original_img.shape[1] / resized_img.shape[1]

        for bbox, class_name, conf in zip(bboxes, class_names, confidences):
            # Scale bounding boxes back to the original image size
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * width_scale)
            y1 = int(y1 * height_scale)
            x2 = int(x2 * width_scale)
            y2 = int(y2 * height_scale)

            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                original_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        return original_img

    def save_image_with_detections(self, img, output_path):
        # Save the image with detections
        Image.fromarray(img).save(output_path)

    def detect_and_save(self, output_path):
        # Capture the screen
        screen_image = self.capture_screen()

        # Convert the captured screen from BGR to RGB
        screen_image = cv2.cvtColor(screen_image, cv2.COLOR_BGR2RGB)

        # Resize image to the input size of the model
        resized_image = self.resize_image(screen_image, self.input_size)

        # Run inference
        results = self.run_inference(resized_image)

        # Draw detections
        output_image = self.draw_detections(screen_image, resized_image, results)

        # Convert the image back to RGB before saving
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        detections = self.get_mobs_bbox(results.pred[0], resized_image, screen_image)

        # Save the annotated image
        self.save_image_with_detections(output_image, output_path)

        return detections

    def run_yolo_inference(self, frame):

        # stride, names, pt = self.model.stride, self.model.names, self.model.pt
        # imgsz = (640, 384)
        # self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *imgsz))
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        # frame.show()
        img = transform(frame).to(self.device)
        # print(img.shape)
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = letterbox(img, new_shape=(384, 640))[0]  # Resize image to 640x640
        img = np.ascontiguousarray(img)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # self.model = self.model.eval()
        if self.verbose:
            print(img.shape)
            print(frame.size)
        # Inference
        with torch.no_grad():
            pred: Detections = self.model(img)
            pred = pred[0]
            from torchvision.ops.boxes import nms

            if pred.shape[0] > 0:
                pred = pred[nms(pred[:, :4], pred[:, 4], iou_threshold=0.1)]
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.size).round()
            # print(f"After NMS {pred}")
            if self.verbose:
                print(pred.shape)
            annotator = Annotator(frame, line_width=3, example=str(self.model.names))

            names = self.model.names
            box_predictions = pred[:, :4]  # Bounding box coordinates

            # print(f"max objectness {torch.argmax(objectness)}")
            class_probs = pred[:, 5:]  # Class probabilities
            max_probs, class_indices = torch.max(class_probs, dim=1)
            if self.verbose:
                print(f"max_probs {max_probs}, {torch.max(max_probs)}")
                print(f"class_indices {class_indices}")
                print(f"shape {class_indices.shape}")
            confidences = pred[:, 4]
            if self.verbose:
                print(f"confidences {confidences}, {torch.max(confidences)}")
            # print(f"class_probs {torch.argmax(class_probs, dim=1)}")
            # Apply objectness score threshold
            objectness_threshold = 0.001
            mask = confidences > objectness_threshold
            filtered_boxes = box_predictions[mask]
            filtered_scores = confidences[mask]
            filtered_class_indices = class_indices[mask]
            filtered_class_probs = class_probs[mask]
            detections = torch.cat(
                (
                    filtered_boxes,
                    filtered_scores.unsqueeze(1),
                    filtered_class_indices.unsqueeze(1).float(),
                ),
                dim=1,
            )
            # Apply NMS (assuming you have defined a function for NMS)
            from torchvision.ops import nms

            # Convert boxes to the format expected by torchvision.ops.nms
            # (x1, y1, x2, y2) and scale coordinates accordingly
            boxes = (
                filtered_boxes  # Assuming these are already scaled to image dimensions
            )
            scores = filtered_scores

            # You might need to adjust the box format based on your model output
            # For example: boxes = xywh2xyxy(filtered_boxes) if the output is in (x, y, w, h) format

            # Apply NMS
            # iou_threshold = 0.1  # Intersection-over-union threshold for NMS
            # keep_indices = nms(boxes, scores, iou_threshold)

            # # Final detections
            # final_boxes = boxes[keep_indices]
            # final_scores = scores[keep_indices]
            # final_class_probs = filtered_class_probs[keep_indices]

            # Convert class probabilities to class labels
            # final_labels = final_class_probs.argmax(dim=1)

            # # Combine results
            # final_detections = torch.cat(
            #     (
            #         final_boxes,
            #         final_scores.unsqueeze(1),
            #         final_labels.unsqueeze(1).float(),
            #     ),
            #     dim=1,
            # )
            if self.verbose:
                print(f"Detections: {detections}")
        return detections
        # detections = []
        # pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.size).round()

        # for det in pred:
        #     if len(det):
        #         # Rescale boxes from img_size to frame size
        #         print(det)

        #         bbox = det[:4]
        #         x1, y1, x2, y2 = map(int, bbox)
        #         conf = det[4]
        #         cls_probs = det[5:]
        #         # det[:, :4] = scale_coords(img.shape[2:], det[:4], img.shape).round()
        #         # for *xyxy, conf, cls in det:
        #         # x1, y1, x2, y2 = map(int, xyxy)
        #         label = int(cls)
        #         confidence = float(conf)
        #         detections.append(
        #             (
        #                 np.array([x1, y1, x2, y2], dtype=np.float32),
        #                 label,
        #                 np.array(confidence, dtype=np.float32),
        #             )
        #         )
        # return detections

    def is_done(self, state):
        if state["health"] <= 0:
            return True
        return False


if __name__ == "__main__":
    register(
        id="RiskOfRain-v0",
        entry_point="RiskOfRainEnv",  # Adjust the entry point if defined in another file/module
    )
    print(gym.spec("RiskOfRain-v0"))
