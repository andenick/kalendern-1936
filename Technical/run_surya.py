"""Surya smoke pipeline for Kalendern 1936 pages (surya-ocr 0.17.1, torch-native).

NOTE on version: surya-ocr 0.20.0 ("Surya 2") dropped in-process torch
inference -- it is now a *client* to an external VLM server (vLLM-in-Docker, or
a spawned `llama-server`). Neither is permitted/available here (no Docker; the
task forbids launching llama-server). 0.17.1 is the last fully torch-native
release: it runs layout + reading-order + text recognition entirely in-process
on the GPU (sm_120 via torch 2.11+cu128). It is the same Surya model family,
exercised programmatically -- exactly what the pipeline needs.

API (surya-ocr 0.17.x):
  from surya.foundation import FoundationPredictor
  from surya.layout import LayoutPredictor
  from surya.recognition import RecognitionPredictor
  from surya.detection import DetectionPredictor
  from surya.settings import settings

  layout_fp = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
  layout_pred = LayoutPredictor(layout_fp)
  rec_fp = FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
  rec_pred = RecognitionPredictor(rec_fp)
  det_pred = DetectionPredictor()

  layouts = layout_pred([img])                       # List[LayoutResult]
  # LayoutResult.bboxes: LayoutBox(polygon, confidence, label, position)
  #   label in {Text, PageHeader, PageFooter, SectionHeader, Picture,
  #             Table, ListItem, Figure, Caption, ...}
  #   position = reading-order index (0,1,2,... in detected reading order)
  ocr = rec_pred([img], det_predictor=det_pred,
                 task_names=["ocr_with_boxes"])       # List[OCRResult]
  # OCRResult.text_lines: TextLine(polygon, confidence, text)
"""
import os, sys, json

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
sys.stdout.reconfigure(encoding="utf-8")

import torch
from PIL import Image
from surya.foundation import FoundationPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.settings import settings

import os
IN_DIR = os.path.join(os.environ.get("DATA_ROOT", "data"), "Inputs", "Images")
OUT_DIR = os.path.join(os.environ.get("OUTPUT_ROOT", "outputs"), "regions")
PAGES = ["SSA_0001", "SSA_0052", "SSA_0103", "SSA_0208", "SSA_0311"]


def poly_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def assign_line_to_region(line_bbox, regions):
    """Return reading-order position of the layout region whose bbox contains
    the line's center, else None. regions = list of dicts with bbox+position."""
    cx = (line_bbox[0] + line_bbox[2]) / 2
    cy = (line_bbox[1] + line_bbox[3]) / 2
    best = None
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            area = (x1 - x0) * (y1 - y0)
            if best is None or area < best[0]:
                best = (area, r)
    return best[1] if best else None


def main():
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
          "| device", settings.TORCH_DEVICE_MODEL, flush=True)

    print("Loading layout predictor...", flush=True)
    layout_fp = FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    layout_pred = LayoutPredictor(layout_fp)

    print("Loading recognition predictor...", flush=True)
    rec_fp = FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
    rec_pred = RecognitionPredictor(rec_fp)

    print("Loading detection predictor...", flush=True)
    det_pred = DetectionPredictor()

    for page in PAGES:
        img_path = os.path.join(IN_DIR, page + ".jpg")
        print(f"\n=== {page} ===", flush=True)
        img = Image.open(img_path).convert("RGB")
        print("  size", img.size, flush=True)

        # --- layout + reading order ---
        layouts = layout_pred([img])
        layout = layouts[0]

        regions = []
        for b in sorted(layout.bboxes,
                        key=lambda x: (x.position if x.position is not None else 9999)):
            regions.append({
                "label": b.label,
                "position": b.position,                # reading-order index
                "confidence": round(float(b.confidence), 4) if b.confidence is not None else None,
                "bbox": [round(v, 1) for v in poly_bbox(b.polygon)],
                "polygon": [[round(p[0], 1), round(p[1], 1)] for p in b.polygon],
            })

        # --- text recognition (line-level, region-aware via internal detection) ---
        ocr = rec_pred([img], det_predictor=det_pred, task_names=["ocr_with_boxes"])
        page_ocr = ocr[0]

        # Map each recognized line to its layout region + tag lines with region pos
        text_lines = []
        for tl in page_ocr.text_lines:
            lb = poly_bbox(tl.polygon)
            reg = assign_line_to_region(lb, regions)
            text_lines.append({
                "text": tl.text,
                "confidence": round(float(tl.confidence), 4) if tl.confidence is not None else None,
                "bbox": [round(v, 1) for v in lb],
                "region_position": reg["position"] if reg else None,
                "region_label": reg["label"] if reg else None,
            })

        # --- write layout JSON ---
        label_counts = {}
        for r in regions:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        layout_out = {
            "page": page,
            "surya_version": "0.17.1",
            "image_size": list(img.size),
            "image_bbox": layout.image_bbox,
            "n_regions": len(regions),
            "label_counts": label_counts,
            "reading_order": [
                {"position": r["position"], "label": r["label"], "bbox": r["bbox"]}
                for r in regions
            ],
            "regions": regions,
            "n_text_lines": len(text_lines),
            "text_lines": text_lines,
        }
        with open(os.path.join(OUT_DIR, f"{page}_surya_layout.json"), "w", encoding="utf-8") as f:
            json.dump(layout_out, f, ensure_ascii=False, indent=2)

        # --- write recognized text, grouped by region in reading order ---
        # Group lines by region position; regions already in reading order.
        out_lines = []
        by_region = {}
        for tl in text_lines:
            by_region.setdefault(tl["region_position"], []).append(tl)
        for r in regions:
            pos = r["position"]
            out_lines.append(f"===== [region {pos}] <{r['label']}> conf={r['confidence']} bbox={r['bbox']} =====")
            rlines = by_region.get(pos, [])
            # order lines within region top-to-bottom then left-to-right
            rlines.sort(key=lambda t: (round(t["bbox"][1] / 10), t["bbox"][0]))
            for tl in rlines:
                if tl["text"].strip():
                    out_lines.append(tl["text"])
            out_lines.append("")
        # any lines that matched no region (orphans)
        orphans = by_region.get(None, [])
        if orphans:
            out_lines.append("===== [unassigned lines] =====")
            orphans.sort(key=lambda t: (round(t["bbox"][1] / 10), t["bbox"][0]))
            for tl in orphans:
                if tl["text"].strip():
                    out_lines.append(tl["text"])
        with open(os.path.join(OUT_DIR, f"{page}_surya_text.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines))

        print(f"  regions={len(regions)} labels={label_counts}", flush=True)
        print(f"  text_lines={len(text_lines)}", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
