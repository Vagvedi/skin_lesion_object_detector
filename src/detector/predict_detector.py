import random
from pathlib import Path

import cv2
import torch

from PIL import Image
from torchvision import transforms

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


# =====================================
# Configuration
# =====================================

TEST_FOLDER = Path(
    "data/multiclass/processed/test/images"
)


MODEL_PATH = (
    "models/detector/"
    "faster_rcnn_multiclass_v2_best.pth"
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


CLASS_NAMES = {

    1: "Benign",
    2: "Malignant",
    3: "AK",
    4: "BCC",
    5: "Dermatofibroma",
    6: "Melanoma",
    7: "Nevus",
    8: "Pigmented BK",
    9: "SCC",
    10: "Vascular"

}



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
    11
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


print("Model Loaded Successfully ✅")



# =====================================
# Select Image
# =====================================


images = (
    list(TEST_FOLDER.glob("*.jpg"))
    +
    list(TEST_FOLDER.glob("*.png"))
    +
    list(TEST_FOLDER.glob("*.jpeg"))
)


if len(images) == 0:

    print(
        "No images found in:",
        TEST_FOLDER
    )

    exit()



image_path = random.choice(
    images
)


print(
    "Testing:",
    image_path.name
)



# =====================================
# Prepare Image
# =====================================


image = Image.open(
    image_path
).convert("RGB")


transform = transforms.ToTensor()


tensor = transform(
    image
).to(DEVICE)



# =====================================
# Prediction
# =====================================


with torch.no_grad():

    prediction = model(
        [tensor]
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



# =====================================
# Draw Results
# =====================================


img = cv2.imread(
    str(image_path)
)


CONF_THRESHOLD = 0.5


detections = 0


for box, score, label in zip(
    boxes,
    scores,
    labels
):

    if score < CONF_THRESHOLD:
        continue


    detections += 1


    x1,y1,x2,y2 = map(
        int,
        box
    )


    class_name = CLASS_NAMES.get(
        int(label),
        "Unknown"
    )


    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )


    cv2.putText(
        img,
        f"{class_name} {score:.2f}",
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )



print(
    "Objects Detected:",
    detections
)



# =====================================
# Display
# =====================================


cv2.imshow(
    "Prediction",
    img
)


cv2.waitKey(0)

cv2.destroyAllWindows()