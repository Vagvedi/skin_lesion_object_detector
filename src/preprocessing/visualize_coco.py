import json
import random
from pathlib import Path

import cv2

# -----------------------------
# Paths
# -----------------------------

BASE = Path("data/detection")

IMAGE_DIR = BASE / "train" / "images"
JSON_FILE = BASE / "annotations" / "train.json"

# -----------------------------
# Load COCO
# -----------------------------

with open(JSON_FILE) as f:
    coco = json.load(f)

# Build image lookup
images = {img["id"]: img for img in coco["images"]}

# Group annotations by image
annotations = {}

for ann in coco["annotations"]:
    annotations.setdefault(ann["image_id"], []).append(ann)

# Pick 5 random images
sample_ids = random.sample(list(images.keys()), 5)

for image_id in sample_ids:

    info = images[image_id]

    img_path = IMAGE_DIR / info["file_name"]

    image = cv2.imread(str(img_path))

    if image is None:
        continue

    for ann in annotations.get(image_id, []):

        x, y, w, h = ann["bbox"]

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        cls = "Benign" if ann["category_id"] == 0 else "Malignant"

        color = (0,255,0) if ann["category_id"] == 0 else (0,0,255)

        cv2.rectangle(image,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            image,
            cls,
            (x1,max(25,y1-10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(info["file_name"], image)

    cv2.waitKey(0)

cv2.destroyAllWindows()