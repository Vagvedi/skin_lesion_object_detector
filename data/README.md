# Data

Images, YOLO labels, and COCO JSON files are not committed. Keep this layout after download and preprocessing:

```
data/
├── raw/                         # original Kaggle sources (per-class folders)
└── multiclass/
    ├── processed/
    │   ├── train/{images,labels}
    │   ├── val/{images,labels}
    │   └── test/{images,labels}
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json
```

Build the processed set from `data/raw/`:

```bash
python src/preprocessing/merge_multiclass_dataset.py
python src/preprocessing/convert_yolo_to_coco.py
python scripts/check_distribution.py
```
