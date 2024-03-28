import os
from time import time
from datetime import datetime
import torch
import ultralytics
from ultralytics import YOLO
from gpus_clearance import clear_gpus_memory

os.environ["OMP_NUM_THREADS"] = "24"

print("Ultralytics version:", ultralytics.__version__)
print("PyTorch version:", torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.version())
print("Available GPU count:", torch.cuda.device_count())


clear_gpus_memory()

# Load a pretrained YOLO model (recommended for training)

name = "test"
# Get the current datetime
now = datetime.now()
# Format the datetime as a string in the format YYYYMMDDHHMMSS
datetime_str = now.strftime("%Y-%m-%d-%H%M%S")

print(os.getcwd())

model = YOLO("models/yolov9c.pt")
print()
print(model.info())
print()

# results = model.train(
#     task="detect",
#     data="/home/ubuntu/datasets/dataset.yaml",
#     epochs=3,
#     imgsz=640,
#     workers=24,
#     batch=64,
#     device=[0, 1, 2, 3],
#     project="runs",
#     name=f"{name}_{datetime_str}",
# )

start_time = time()
# Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
result_grid = model.tune(
    task="detect",
    data="/home/ubuntu/datasets/dataset.yaml",
    space={
        "lr0": (1e-3, 1e-1),
        "lrf": (1e-3, 1e-1),
        "weight_decay": (1e-5, 1e-3),
        "warmup_epochs": (2.0, 4.0),
        "warmup_momentum": (0.6, 0.95),
        "box": (5.0, 15.0),
        "cls": (0.2, 1),
        "dfl": (1, 3),
        "hsv_h": (0.0, 0.04),
        "hsv_s": (0.5, 0.9),
        "hsv_v": (0.3, 0.7),
        "translate": (0.0, 0.3),
        "scale": (0.0, 0.5),
        "fliplr": (0.3, 0.65),
        "mosaic": (0.9, 1.0),
    },
    epochs=30,
    imgsz=640,
    batch=64,
    dropout=0.1,
    project="runs",
    name=f"{name}_{datetime_str}",
    iterations=60,
    device=[0, 1, 2, 3],
    workers=24,
    patience=10,
    plots=True,
    save=False,
    val=True,
    use_ray=False,
)
print(f"\n Overall process took: {(time() - start_time):.3f} sec.")
