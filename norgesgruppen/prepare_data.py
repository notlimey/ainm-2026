"""Convert COCO annotations to YOLO format with train/val split."""
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.15  # ~37 images for validation

def main():
    base = Path(__file__).parent
    coco_path = base / "train" / "annotations.json"
    images_dir = base / "train" / "images"

    with open(coco_path) as f:
        coco = json.load(f)

    # Build lookup: image_id -> image info
    img_lookup = {img["id"]: img for img in coco["images"]}

    # Build lookup: image_id -> list of annotations
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # Category names ordered by id
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_names = [c["name"] for c in categories]
    nc = len(categories)
    print(f"Number of classes: {nc}")
    print(f"Number of images: {len(coco['images'])}")
    print(f"Number of annotations: {len(coco['annotations'])}")

    # Train/val split
    random.seed(SEED)
    image_ids = sorted(img_lookup.keys())
    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * VAL_RATIO))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create output directories
    out = base / "yolo_dataset"
    for split in ["train", "val"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Convert and copy
    for img_id in image_ids:
        img_info = img_lookup[img_id]
        w, h = img_info["width"], img_info["height"]
        fname = img_info["file_name"]
        split = "val" if img_id in val_ids else "train"

        # Copy image (symlink to save space)
        src = images_dir / fname
        dst = out / "images" / split / fname
        if not dst.exists():
            shutil.copy2(src, dst)

        # Write YOLO label file
        label_name = Path(fname).stem + ".txt"
        label_path = out / "labels" / split / label_name
        lines = []
        for ann in img_anns.get(img_id, []):
            cat_id = ann["category_id"]
            bx, by, bw, bh = ann["bbox"]  # COCO: x, y, w, h (top-left)
            # Convert to YOLO: x_center, y_center, w, h (normalized)
            x_center = (bx + bw / 2) / w
            y_center = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")
        label_path.write_text("\n".join(lines) + "\n" if lines else "")

    # Write data.yaml
    yaml_path = base / "data.yaml"
    # Write as plain text to avoid yaml import issues
    yaml_lines = [
        f"path: {out.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
        "names:",
    ]
    for i, name in enumerate(cat_names):
        # Use double quotes to handle names with apostrophes (e.g. KELLOGG'S)
        safe_name = name.replace('"', '\\"')
        yaml_lines.append(f'  {i}: "{safe_name}"')

    yaml_path.write_text("\n".join(yaml_lines) + "\n")
    print(f"Created {yaml_path}")
    print("Done!")


if __name__ == "__main__":
    main()
