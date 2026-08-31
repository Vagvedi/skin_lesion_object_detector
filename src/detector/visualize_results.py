import torch
import cv2
import os
import random
import json

from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


MODEL_PATH = "models/detector/faster_rcnn_resnet50_best.pth"

IMAGE_DIR = "data/detection/test/images"
ANN_FILE = "data/detection/test/annotations/test.json"

SAVE_DIR = "results/evaluation/predictions"


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# -----------------------------
# Load COCO annotations
# -----------------------------

with open(ANN_FILE) as f:
    coco = json.load(f)


annotations = {}

for ann in coco["annotations"]:

    img_id = ann["image_id"]

    if img_id not in annotations:
        annotations[img_id] = []

    annotations[img_id].append(ann)



images_info = coco["images"]


# -----------------------------
# Load Model
# -----------------------------

model = fasterrcnn_resnet50_fpn_v2(
    weights=None
)


num_classes = 3


in_features = (
    model.roi_heads
    .box_predictor
    .cls_score
    .in_features
)


model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)


model.to(DEVICE)

model.eval()


print("Model loaded")


transform = transforms.ToTensor()



# -----------------------------
# Pick 10 random images
# -----------------------------

samples = random.sample(
    images_info,
    10
)


for img_info in samples:


    filename = img_info["file_name"]

    img_id = img_info["id"]


    path = os.path.join(
        IMAGE_DIR,
        filename
    )


    image = cv2.imread(path)

    output = image.copy()



    # -------------------------
    # Ground Truth BOX
    # Blue
    # -------------------------

    for ann in annotations.get(img_id, []):

        x,y,w,h = ann["bbox"]


        cv2.rectangle(
            output,
            (int(x),int(y)),
            (int(x+w),int(y+h)),
            (255,0,0),
            2
        )



    # -------------------------
    # Prediction
    # Green
    # -------------------------

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    tensor = transform(rgb)


    with torch.no_grad():

        pred = model(
            [tensor.to(DEVICE)]
        )


    boxes = pred[0]["boxes"].cpu()
    scores = pred[0]["scores"].cpu()


    for box,score in zip(boxes,scores):

        if score < 0.5:
            continue


        x1,y1,x2,y2 = map(
            int,
            box
        )


        cv2.rectangle(
            output,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.putText(
            output,
            f"{score:.2f}",
            (x1,y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )



    save_path = os.path.join(
        SAVE_DIR,
        filename
    )


    cv2.imwrite(
        save_path,
        output
    )


    print(
        "Saved:",
        filename
    )


print("Visualization completed ✅")