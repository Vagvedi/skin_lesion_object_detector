# Skin Lesion Detection

Multi-class Faster R-CNN (ResNet50 FPN) pipeline: Kaggle sources → CVAT/YOLO labels → COCO conversion → train → COCO evaluation.

Engineering write-up: open [`index.html`](index.html) in a browser.

## Classes

`benign`, `malignant`, `ak`, `bcc`, `dermatofibroma`, `melanoma`, `nevus`, `pigmented_bk`, `scc`, `vascular`

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place source datasets under `data/raw/` (see `data/README.md`). Weights go under `models/` after training (`models/README.md`). Both are gitignored.

## Pipeline

```bash
python src/preprocessing/merge_multiclass_dataset.py
python src/preprocessing/convert_yolo_to_coco.py
python scripts/check_distribution.py
python src/detector/train_detector.py
python src/detector/evaluate_metrics.py
python src/detector/evaluate_classwise.py
```

Run detector scripts from `src/detector/` or ensure that folder is on `PYTHONPATH` so `from dataset import SkinLesionDataset` resolves.

## Layout

```
src/preprocessing/   merge, YOLO→COCO, annotation checks
src/detector/        train, predict, COCO + class-wise eval
scripts/             class-distribution audit
index.html           project documentation
classwise_results.txt test-set recall table
```
