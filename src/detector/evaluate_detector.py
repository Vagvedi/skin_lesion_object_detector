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

VAL_IMAGES = (
    "data/multiclass/"
    "processed/val/images"
)


VAL_ANN = (
    "data/multiclass/"
    "annotations/val.json"
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

print(
    "Faster R-CNN 10-Class Skin Lesion Evaluation"
)

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

    VAL_IMAGES,

    VAL_ANN,

    transform

)


print(
    "Validation Images:",
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
# Evaluation
# =====================================

CONF_THRESHOLD = 0.5



results = {}



for cls in CLASSES[1:]:

    results[cls] = {

        "total":0,

        "detected":0

    }



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



        gt_labels = (

            target["labels"]

            .numpy()

        )



        for label in gt_labels:


            class_name = CLASSES[label]


            results[class_name]["total"] += 1



            if label in pred_labels:


                results[class_name]["detected"] += 1





# =====================================
# Report
# =====================================

print("\n")

print("="*60)

print(
    "CLASS-WISE DETECTION RESULTS"
)

print("="*60)



total_objects = 0

total_detected = 0



for cls,value in results.items():


    total = value["total"]

    detected = value["detected"]



    if total == 0:

        continue



    recall = (

        detected /

        total

    )



    total_objects += total

    total_detected += detected



    print(

        f"{cls:<20}"

        f"Recall: {recall*100:.2f}% "

        f"({detected}/{total})"

    )




overall = (

    total_detected /

    total_objects

)



print("\n")

print("="*60)


print(

    f"Overall Detection Recall: "

    f"{overall*100:.2f}%"

)


print("="*60)