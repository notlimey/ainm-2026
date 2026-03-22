"""NorgesGruppen shelf product detection — ONNX submission."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def letterbox(img, new_shape=640):
    """Resize and pad image to square while maintaining aspect ratio."""
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = np.array(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    padded[top : top + nh, left : left + nw] = img_resized
    return padded, scale, left, top


def postprocess(output, conf_thresh, scale, pad_left, pad_top):
    """Parse YOLOv8 ONNX output [1, 361, 8400] → list of (x1,y1,x2,y2,conf,cls)."""
    # output shape: [1, num_classes+4, num_detections]
    pred = output[0]  # [361, 8400]
    pred = pred.T  # [8400, 361]
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]

    max_scores = scores.max(axis=1)
    mask = max_scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    max_scores = max_scores[mask]
    class_ids = scores.argmax(axis=1)

    results = []
    for i in range(len(boxes_xywh)):
        cx, cy, w, h = boxes_xywh[i]
        x1 = (cx - w / 2 - pad_left) / scale
        y1 = (cy - h / 2 - pad_top) / scale
        w_orig = w / scale
        h_orig = h / scale
        results.append((x1, y1, w_orig, h_orig, float(max_scores[i]), int(class_ids[i])))
    return results


def nms(detections, iou_thresh=0.5):
    """Simple NMS."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        bx1, by1, bw, bh = best[:4]
        for d in dets:
            dx1, dy1, dw, dh = d[:4]
            ix1 = max(bx1, dx1)
            iy1 = max(by1, dy1)
            ix2 = min(bx1 + bw, dx1 + dw)
            iy2 = min(by1 + bh, dy1 + dh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = bw * bh + dw * dh - inter
            if union > 0 and inter / union < iou_thresh:
                remaining.append(d)
        dets = remaining
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = Path(__file__).parent / "best.onnx"
    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    predictions = []
    input_dir = Path(args.input)

    for img_path in sorted(input_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        image_id = int(img_path.stem.split("_")[-1])

        img = np.array(Image.open(img_path).convert("RGB"))
        padded, scale, pad_left, pad_top = letterbox(img, 640)

        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = session.run(None, {input_name: blob})
        dets = postprocess(outputs[0], conf_thresh=0.1, scale=scale, pad_left=pad_left, pad_top=pad_top)
        dets = nms(dets, iou_thresh=0.5)

        for x1, y1, w, h, score, cls_id in dets:
            predictions.append({
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(score, 3),
            })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    main()
