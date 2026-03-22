"""Train YOLOv8m on the NorgesGruppen shelf detection dataset.

Usage:
    pip install ultralytics==8.1.0
    python train_yolo.py

Expects yolo_dataset/ and data.yaml to exist (run prepare_data.py first).
"""
import torch

# Patch: torch >=2.6 defaults weights_only=True, which breaks ultralytics 8.1.0 model loading.
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO

def main():
    # Resume from last checkpoint to continue training
    model = YOLO("runs/yolov8m_ngd/weights/last.pt")

    model.train(
        data="data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,           # TAL patched to CPU fallback, can use larger batch now
        patience=30,       # early stopping
        device="mps",      # Apple Silicon GPU
        workers=4,
        project="runs",
        name="yolov8m_ngd",
        resume=True,       # resume from checkpoint
        # Augmentation — re-enabled now that TAL is patched
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        # Other
        save=True,
        save_period=25,    # save checkpoint every 25 epochs
        val=True,
        plots=True,
        verbose=True,
    )

    # Validate best model
    best = YOLO("runs/yolov8m_ngd/weights/best.pt")
    metrics = best.val(data="data.yaml", imgsz=640)
    print(f"\nBest model mAP50: {metrics.box.map50:.4f}")
    print(f"Best model mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
