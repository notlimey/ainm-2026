"""Evaluate a trained model locally using the same scoring as the competition.

Usage:
    python evaluate.py --model runs/yolov8m_ngd/weights/best.pt

Runs inference on the val split, computes:
  - detection mAP@0.5 (all preds treated as single class)
  - classification mAP@0.5 (category must match)
  - combined score = 0.7 * det + 0.3 * cls
"""
import argparse
import json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import copy

import torch
from ultralytics import YOLO


def run_inference(model_path, images_dir, conf=0.1):
    """Run YOLO inference and return COCO-format predictions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    predictions = []

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])
        with torch.no_grad():
            results = model(str(img_path), device=device, verbose=False, conf=conf)
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(r.boxes.cls[i].item()),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                    "score": round(float(r.boxes.conf[i].item()), 3),
                })
    return predictions


def build_val_coco(annotations_path, val_image_ids):
    """Build a COCO ground truth object for val images only."""
    with open(annotations_path) as f:
        full = json.load(f)

    val_set = set(val_image_ids)
    gt = {
        "images": [img for img in full["images"] if img["id"] in val_set],
        "annotations": [ann for ann in full["annotations"] if ann["image_id"] in val_set],
        "categories": full["categories"],
    }
    # Re-assign annotation ids to be sequential
    for i, ann in enumerate(gt["annotations"]):
        ann["id"] = i + 1

    tmp = Path("/tmp/val_gt.json")
    with open(tmp, "w") as f:
        json.dump(gt, f)
    return COCO(str(tmp))


def compute_map(coco_gt, predictions, ignore_category=False):
    """Compute mAP@0.5. If ignore_category, treat all as single class."""
    if not predictions:
        return 0.0

    preds = copy.deepcopy(predictions)
    gt = copy.deepcopy(coco_gt)

    if ignore_category:
        # Set all categories to 1 for detection-only eval
        for ann_id in gt.anns:
            gt.anns[ann_id]["category_id"] = 1
        for img_id in gt.imgToAnns:
            for ann in gt.imgToAnns[img_id]:
                ann["category_id"] = 1
        for p in preds:
            p["category_id"] = 1
        # Also update catToImgs
        gt.catToImgs = {1: list(gt.imgs.keys())}
        gt.cats = {1: {"id": 1, "name": "product", "supercategory": "product"}}
        gt.dataset["categories"] = [{"id": 1, "name": "product", "supercategory": "product"}]

    coco_dt = gt.loadRes(preds)
    coco_eval = COCOeval(gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]  # only IoU=0.5
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]  # mAP@0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    args = parser.parse_args()

    base = Path(__file__).parent
    val_images = base / "yolo_dataset" / "images" / "val"
    annotations = base / "train" / "annotations.json"

    # Get val image IDs
    val_image_ids = []
    for p in val_images.iterdir():
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            val_image_ids.append(int(p.stem.split("_")[-1]))

    print(f"Running inference on {len(val_image_ids)} val images...")
    predictions = run_inference(args.model, val_images, conf=args.conf)
    print(f"Generated {len(predictions)} predictions")

    # Filter predictions to val images only
    val_set = set(val_image_ids)
    predictions = [p for p in predictions if p["image_id"] in val_set]

    # Build ground truth
    coco_gt = build_val_coco(annotations, val_image_ids)

    # Detection mAP (category ignored)
    print("\n=== Detection mAP@0.5 (category ignored) ===")
    det_map = compute_map(coco_gt, predictions, ignore_category=True)

    # Classification mAP (category must match)
    coco_gt2 = build_val_coco(annotations, val_image_ids)
    print("\n=== Classification mAP@0.5 (category must match) ===")
    cls_map = compute_map(coco_gt2, predictions, ignore_category=False)

    # Combined score
    score = 0.7 * det_map + 0.3 * cls_map
    print(f"\n{'='*50}")
    print(f"Detection mAP@0.5:       {det_map:.4f} (x0.7 = {0.7*det_map:.4f})")
    print(f"Classification mAP@0.5:  {cls_map:.4f} (x0.3 = {0.3*cls_map:.4f})")
    print(f"Combined Score:          {score:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
