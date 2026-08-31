import torch
from tqdm import tqdm
from torchvision import transforms

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

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



print("="*70)

print(
    "Class Wise Detection Recall Evaluation"
)

print("="*70)


print(
    "Device:",
    DEVICE
)



# =====================================
# Dataset
# =====================================

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)


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
# Evaluation Setup
# =====================================


CONF_THRESHOLD = 0.5



results = {}



for cls in CLASSES[1:]:

    results[cls] = {

        "detected":0,

        "total":0

    }



# =====================================
# Prediction
# =====================================


with torch.no_grad():


    for idx in tqdm(
        range(len(dataset)),
        desc="Evaluating"
    ):


        image, target = dataset[idx]


        image = image.to(
            DEVICE
        )



        prediction = model(
            [image]
        )[0]



        pred_labels = (
            prediction["labels"]
            .cpu()
            .numpy()
        )


        pred_scores = (
            prediction["scores"]
            .cpu()
            .numpy()
        )



        pred_labels = pred_labels[
            pred_scores >= CONF_THRESHOLD
        ]



        true_labels = (
            target["labels"]
            .numpy()
        )



        for label in true_labels:


            class_name = CLASSES[label]


            results[class_name]["total"] += 1



            if label in pred_labels:

                results[class_name]["detected"] += 1





# =====================================
# Final Report
# =====================================


output = []


output.append("\n")
output.append("="*70)

output.append(
    "Model Evaluation Results - Faster R-CNN ResNet50"
)

output.append("="*70)



header = (
    f"{'Disease Class':30}"
    f"{'Detected':15}"
    f"{'Total':15}"
    f"{'Recall (%)'}"
)


output.append(header)

output.append("-"*70)



for cls,value in results.items():


    total = value["total"]

    detected = value["detected"]



    if total == 0:

        continue



    recall = (
        detected /
        total
    ) * 100



    line = (

        f"{cls:30}"

        f"{detected:<15}"

        f"{total:<15}"

        f"{recall:.2f}"

    )


    output.append(line)




print("\n")


for line in output:

    print(line)



# =====================================
# Save Report
# =====================================


with open(
    "classwise_results.txt",
    "w"
) as f:

    for line in output:

        f.write(
            line + "\n"
        )



print("\n")

print(
    "Class-wise report saved ✅"
)

print(
    "File: classwise_results.txt"
)