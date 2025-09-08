# annotator.py
"""
Simple Tkinter-based annotator that opens an image, lets user draw bounding boxes,
enter class name (or choose existing), and saves YOLO-format label.
This is intentionally minimal—meant for quick in-class data collection.
"""
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import os
from utils import ensure_dirs
import json

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
ensure_dirs(IMAGE_DIR, LABEL_DIR)

class Annotator:
    def __init__(self, master):
        self.master = master
        self.master.title("Annotator")
        self.canvas = tk.Canvas(master, cursor="cross")
        self.canvas.pack(fill="both", expand=True)
        self.img = None
        self.tkimg = None
        self.image_path = None
        self.bbox_start = None
        self.current_rect = None
        self.boxes = []
        self.classes = []  # dynamic classes list; persisted optionally

        menubar = tk.Menu(master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Image...", command=self.open_image)
        filemenu.add_command(label="Save Labels", command=self.save_labels)
        menubar.add_cascade(label="File", menu=filemenu)
        master.config(menu=menubar)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def open_image(self):
        path = filedialog.askopenfilename(title="Open image", filetypes=[("Images","*.jpg *.png *.jpeg")])
        if not path:
            return
        self.image_path = path
        pil = Image.open(path)
        self.img_w, self.img_h = pil.size
        self.img = pil
        self.tkimg = ImageTk.PhotoImage(pil)
        self.canvas.config(width=self.img_w, height=self.img_h)
        self.canvas.create_image(0,0, anchor="nw", image=self.tkimg)
        self.boxes = []

    def on_press(self, event):
        self.bbox_start = (event.x, event.y)
        self.current_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red", width=2)

    def on_drag(self, event):
        if self.current_rect:
            x0,y0 = self.bbox_start
            self.canvas.coords(self.current_rect, x0, y0, event.x, event.y)

    def on_release(self, event):
        if not self.current_rect:
            return
        x0,y0 = self.bbox_start
        x1,y1 = event.x, event.y
        # normalize x0<->x1
        x0n, x1n = sorted([x0,x1])
        y0n, y1n = sorted([y0,y1])
        cls_name = simpledialog.askstring("Class", "Enter class name (e.g., apple):")
        if not cls_name:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
            return
        # persist class to classes list
        if cls_name not in self.classes:
            self.classes.append(cls_name)
        # save box (absolute)
        self.boxes.append({"class": cls_name, "box": [x0n, y0n, x1n, y1n]})
        # draw label
        self.canvas.create_text(x0n+10, y0n+10, text=cls_name, anchor="nw", fill="yellow")
        self.current_rect = None

    def save_labels(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded")
            return
        basename = os.path.splitext(os.path.basename(self.image_path))[0]
        # copy image to data/images if not already
        dest_img = os.path.join(IMAGE_DIR, os.path.basename(self.image_path))
        if not os.path.exists(dest_img):
            from shutil import copyfile
            copyfile(self.image_path, dest_img)
        # save YOLO label file normalized
        label_path = os.path.join(LABEL_DIR, f"{basename}.txt")
        img_w, img_h = self.img.size
        with open(label_path, "w", encoding="utf-8") as f:
            for b in self.boxes:
                cls_name = b["class"]
                # map classes to indices (simple local mapping)
                cls_idx = self.classes.index(cls_name)
                x1,y1,x2,y2 = b["box"]
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        messagebox.showinfo("Saved", f"Saved labels to {label_path}\nClasses: {self.classes}")
        # Save classes mapping (simple)
        with open(os.path.join(LABEL_DIR, "classes.json"), "w", encoding="utf-8") as cf:
            json.dump(self.classes, cf, indent=2)
