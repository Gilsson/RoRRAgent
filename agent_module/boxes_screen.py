import tkinter as tk
from ctypes import windll
import sys


class BoundingBoxOverlay:
    def __init__(self):
        # Create a transparent, non-usable window
        self.root = tk.Tk()
        self.root.title("Overlay")

        # Make the window fullscreen and remove borders
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)  # Keep window on top of all others
        self.root.attributes("-transparentcolor", "blue")  # Set 'blue' as transparent

        # Set background color to blue (this will be transparent)
        self.root.config(bg="blue")

        # Make window non-usable/click-through (Windows specific)
        hwnd = windll.user32.GetForegroundWindow()
        windll.user32.SetWindowLongW(
            hwnd, -20, windll.user32.GetWindowLongW(hwnd, -20) | 0x80000 | 0x20
        )

        # Create a canvas for drawing
        self.canvas = tk.Canvas(self.root, bg="blue", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize an empty list for bounding boxes
        self.bounding_boxes = []  # {"bbox": [], "class": 0, "confidence": 0}

        # Gracefully exit when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start the drawing process
        self.draw_bounding_boxes()

    # Function to update the bounding boxes
    def update_bounding_boxes(self, new_boxes):
        self.bounding_boxes = new_boxes
        self.draw_bounding_boxes()

    # Function to draw bounding boxes
    def draw_bounding_boxes(self):
        self.canvas.delete("all")  # Clear previous drawings
        for box in self.bounding_boxes:
            x1, y1, x2, y2 = box["bbox"]
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=3)
            self.canvas.create_text(x1, y1, text=box["class"], fill="green")
            self.canvas.create_text(x1, y2, text=box["confidence"], fill="green")
        # self.root.after(100, self.draw_bounding_boxes)  # Refresh every 100ms

    # Gracefully close the window
    def on_closing(self):
        print("Closing the window gracefully.")
        self.root.destroy()
        sys.exit(0)

    # Start the Tkinter main loop
    def run(self):
        self.draw_bounding_boxes()
        self.root.mainloop()
