# trainer.py
"""
Trainer wrapper for YOLOv12 (ultralytics) or a stub guidance if not available.
This is intentionally simple: it creates a training YAML (ultralytics style)
and calls the YOLO.train API if ultralytics is installed.
"""
import os
import json
from utils import load_config
cfg = load_config()

HAS_ULTRALYTICS = False
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

def prepare_dataset_structure(data_dir="data", classes=None):
    # ultralytics expects a dataset YAML with train/val image paths; for small classroom use we can use same set
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    # Build dataset YAML content
    data_yaml = {
        "path": data_dir,
        "train": "images",
        "val": "images",
        "names": classes or []
    }
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    import yaml
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f)
    return yaml_path

def _normalize_model_size(name: str) -> str:
    """Map requested model size to a supported Ultralytics model string.
    Fallback order: explicit name -> yolo11n -> yolov8n.
    Also remap unsupported future tokens (e.g., yolov12*) to yolo11 equivalents.
    """
    if not name:
        return "yolo11n"
    # If user requested non-existing 'yolov12*', downgrade to 'yolo11*'
    if name.startswith("yolov12"):
        name = name.replace("yolov12", "yolo11", 1)
    # Accept direct .pt paths
    if name.endswith('.pt') and os.path.isfile(name):
        return name
    # Basic whitelist of common sizes
    allowed_prefixes = ["yolo11", "yolov8"]
    if any(name.startswith(p) for p in allowed_prefixes):
        return name
    # Fallback
    return "yolo11n"

def train(model_size=None, epochs=None, batch_size=None, data_dir="data"):
    raw_model_size = model_size or cfg.get("model_size", "yolo11n")
    model_size = _normalize_model_size(raw_model_size)
    if raw_model_size != model_size:
        print(f"[trainer] Requested model '{raw_model_size}' not found; using '{model_size}' instead.")
    epochs = epochs or cfg.get("train_epochs", 20)
    # Config still stores 'batch_size'; Ultralytics expects arg name 'batch'.
    batch = batch_size or cfg.get("batch_size", 8)
    # Load classes from classes.json if present (preferred)
    classes = []
    labels_dir = os.path.join(data_dir, "labels")
    classes_path = os.path.join(labels_dir, "classes.json")
    if os.path.exists(classes_path):
        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                classes = json.load(f)
        except Exception:
            classes = cfg.get("classes", [])
    else:
        classes = cfg.get("classes", [])
    if not classes:
        print("No classes found. Add objects before training.")
        return False
    # Quick sanity: ensure at least one label file exists (excluding classes.json)
    label_files = []
    if os.path.isdir(labels_dir):
        try:
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'classes.json']
        except Exception as e:
            print(f"[trainer] Failed to list labels dir: {e}")
    if not label_files:
        print("No label files present in labels directory; aborting training.")
        return False

    # Sanitize labels (fix common issues like out-of-range class indices)
    _sanitize_label_files(labels_dir, classes)
    if not HAS_ULTRALYTICS:
        print("Ultralytics not installed. To train, install ultralytics per README.")
        print("You can still test the app with pre-trained models or stub data.")
        return False
    yaml_path = prepare_dataset_structure(data_dir=data_dir, classes=classes)
    # instantiate a model (if model file provided, you could pass pretrained weights)
    try:
        model = YOLO(model_size)  # will download base weights if needed
    except Exception as e:
        # Final fallback cascade
        fallback_list = ["yolo11n", "yolov8n"]
        model = None
        for cand in fallback_list:
            try:
                print(f"[trainer] Attempting fallback model '{cand}'...")
                model = YOLO(cand)
                print(f"[trainer] Using fallback model '{cand}'.")
                break
            except Exception:
                continue
        if model is None:
            print(f"[trainer] Failed to load any YOLO model (last error: {e}). Aborting training.")
            return False
    print("Starting training... this may take time. Check progress in console.")
    # Ultralytics API: use 'batch' not 'batch_size'
    model.train(data=yaml_path, epochs=epochs, batch=batch, imgsz=cfg.get("img_size",640))
    # Save best weights path recommended by ultralytics is run/...
    print("Training complete. Please move the best weights to models/best.pt or update config.")
    return True

def _sanitize_label_files(labels_dir: str, classes):
    """Validate and repair YOLO label txt files.
    Fixes scenario: only one class defined but labels use index 1 (off-by-one) -> remap to 0.~
    Removes lines with malformed numbers or out-of-range indices (unless single-class remap applies).
    """
    if not os.path.isdir(labels_dir):
        return
    class_count = len(classes)
    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt') or fname == 'classes.json':
            continue
        path = os.path.join(labels_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[trainer] Could not read label file {fname}: {e}")
            continue
        changed = False
        valid_out = []
        for ln in lines:
            parts = ln.strip().split()
            if len(parts) != 5:
                print(f"[trainer] Skipping malformed line in {fname}: {ln.strip()}")
                changed = True
                continue
            try:
                cls_idx = int(parts[0])
                vals = list(map(float, parts[1:]))
            except ValueError:
                print(f"[trainer] Non-numeric values in {fname}: {ln.strip()}")
                changed = True
                continue
            # Off-by-one fix: only one class defined but label uses index 1
            if class_count == 1 and cls_idx == 1:
                print(f"[trainer] Remapping class index 1->0 in {fname}")
                cls_idx = 0
                changed = True
            if cls_idx < 0 or cls_idx >= class_count:
                print(f"[trainer] Dropping line with out-of-range class {cls_idx} in {fname}")
                changed = True
                continue
            # Basic range check for normalized coords
            cx, cy, bw, bh = vals
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                print(f"[trainer] Dropping line with invalid normalized coords in {fname}: {ln.strip()}")
                changed = True
                continue
            valid_out.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        if not valid_out:
            print(f"[trainer] No valid labels remain in {fname}; deleting file.")
            try:
                os.remove(path)
            except Exception:
                pass
            continue
        if changed:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.writelines(valid_out)
                print(f"[trainer] Rewrote sanitized labels for {fname}")
            except Exception as e:
                print(f"[trainer] Failed to write sanitized labels for {fname}: {e}")

if __name__ == "__main__":
    train()
