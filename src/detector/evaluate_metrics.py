import torch
import json
import os

from tqdm import tqdm

from torchvision import transforms

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import SkinLesionDataset



# =====================================
# Paths
# =====================================

TEST_IMAGES = (
    "data/multiclass/"
    "processed/test/images"
)


TEST_ANN = (
    "data/multiclass/"
    "annotations/test.json"
)


MODEL_PATH = (
    "models/detector/"
    "faster_rcnn_multiclass_v2_best.pth"
)



# =====================================
# Classes
# =====================================

CLASSES = [

    "background",

    "benign",

    "malignant",

    "ak",

    "bcc",

    "dermatofibroma",

    "melanoma",

    "nevus",

    "pigmented_bk",

    "scc",

    "vascular"

]


NUM_CLASSES = len(CLASSES)



# =====================================
# Device
# =====================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("="*60)
print("Faster R-CNN Detection Metrics Evaluation")
print("="*60)

print(
    "Device:",
    DEVICE
)



# =====================================
# Dataset
# =====================================

transform = transforms.Compose([
    transforms.ToTensor()
])


dataset = SkinLesionDataset(
    TEST_IMAGES,
    TEST_ANN,
    transform
)


print(
    "Test Images:",
    len(dataset)
)



# =====================================
# Load Model
# =====================================

model = fasterrcnn_resnet50_fpn_v2(
    weights=None
)


in_features = (
    model.roi_heads
    .box_predictor
    .cls_score
    .in_features
)


model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)



checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint
)


model.to(DEVICE)

model.eval()


print(
    "Model Loaded Successfully ✅"
)



# =====================================
# Generate Predictions
# =====================================

results = []


CONF_THRESHOLD = 0.05


with torch.no_grad():


    for idx in tqdm(
        range(len(dataset)),
        desc="Generating Predictions"
    ):


        image, target = dataset[idx]


        image = image.to(
            DEVICE
        )


        prediction = model(
            [image]
        )[0]



        boxes = (
            prediction["boxes"]
            .cpu()
            .numpy()
        )


        scores = (
            prediction["scores"]
            .cpu()
            .numpy()
        )


        labels = (
            prediction["labels"]
            .cpu()
            .numpy()
        )



        image_id = (
            target["image_id"]
            .item()
        )



        for box,score,label in zip(
            boxes,
            scores,
            labels
        ):


            if score < CONF_THRESHOLD:
                continue



            x1,y1,x2,y2 = box


            results.append(

                {

                "image_id": int(image_id),

                "category_id": int(label)-1,

                "bbox":[

                    float(x1),

                    float(y1),

                    float(x2-x1),

                    float(y2-y1)

                ],

                "score":float(score)

                }

            )



# =====================================
# Save Predictions
# =====================================

with open(
    "prediction_results.json",
    "w"
) as f:

    json.dump(
        results,
        f
    )


print(
    "\nPredictions saved ✅"
)



# =====================================
# COCO Evaluation
# =====================================

coco_gt = COCO(
    TEST_ANN
)


coco_dt = coco_gt.loadRes(
    "prediction_results.json"
)



coco_eval = COCOeval(
    coco_gt,
    coco_dt,
    "bbox"
)



coco_eval.evaluate()

coco_eval.accumulate()

coco_eval.summarize()



print("\n========== FINAL METRICS ==========")


print(
    f"mAP@0.5:0.95  : {coco_eval.stats[0]:.4f}"
)


print(
    f"mAP@0.5       : {coco_eval.stats[1]:.4f}"
)


print(
    f"mAP@0.75      : {coco_eval.stats[2]:.4f}"
)


print(
    f"AR@100        : {coco_eval.stats[8]:.4f}"
)


print(
    "=================================="
)