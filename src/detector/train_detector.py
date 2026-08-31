import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from tqdm import tqdm

from dataset import SkinLesionDataset


# =====================================
# Configuration
# =====================================

TRAIN_IMAGES = (
    "data/multiclass/processed/train/images"
)

TRAIN_ANN = (
    "data/multiclass/annotations/train.json"
)


VAL_IMAGES = (
    "data/multiclass/processed/val/images"
)

VAL_ANN = (
    "data/multiclass/annotations/val.json"
)


MODEL_PATH = (
    "models/detector/"
    "faster_rcnn_multiclass_v2_best.pth"
)


EPOCHS = 80

BATCH_SIZE = 2

LR = 0.001



DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



CLASS_NAMES = [

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



print("=" * 60)

print(
    "Faster R-CNN ResNet50 FPN "
    "10-Class Skin Lesion Detector V2"
)

print("=" * 60)


print(
    "Device:",
    DEVICE
)


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )



print("\nClasses:")

for i,name in enumerate(CLASS_NAMES):

    print(
        i,
        "->",
        name
    )



# =====================================
# Transform
# =====================================

transform = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor()

])



# =====================================
# Collate
# =====================================

def collate_fn(batch):

    return tuple(zip(*batch))



# =====================================
# Dataset
# =====================================

train_dataset = SkinLesionDataset(

    TRAIN_IMAGES,

    TRAIN_ANN,

    transform

)



val_dataset = SkinLesionDataset(

    VAL_IMAGES,

    VAL_ANN,

    transform

)



train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn,

    num_workers=0,

    pin_memory=True

)



val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_fn,

    num_workers=0,

    pin_memory=True

)



print()

print(
    "Training Images:",
    len(train_dataset)
)


print(
    "Validation Images:",
    len(val_dataset)
)



# =====================================
# Model
# =====================================

model = fasterrcnn_resnet50_fpn_v2(

    weights="DEFAULT"

)



# background + 10 diseases

num_classes = 11



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



model.to(DEVICE)



print(

    "Model Device:",

    next(model.parameters()).device

)



# =====================================
# Optimizer
# =====================================

optimizer = torch.optim.SGD(

    model.parameters(),

    lr=LR,

    momentum=0.9,

    weight_decay=0.0005

)



# Learning rate scheduler

scheduler = torch.optim.lr_scheduler.StepLR(

    optimizer,

    step_size=25,

    gamma=0.1

)



# =====================================
# AMP
# =====================================

use_amp = torch.cuda.is_available()



if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None



# =====================================
# Training
# =====================================

best_loss = float("inf")



for epoch in range(EPOCHS):


    model.train()


    total_loss = 0



    print("\n")

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )



    progress = tqdm(train_loader)



    for images, targets in progress:


        images = [

            img.to(
                DEVICE,
                non_blocking=True
            )

            for img in images

        ]



        targets = [

            {

                k:v.to(

                    DEVICE,

                    non_blocking=True

                )

                for k,v in target.items()

            }

            for target in targets

        ]



        optimizer.zero_grad()



        if use_amp:


            with torch.amp.autocast(
                "cuda"
            ):


                loss_dict = model(

                    images,

                    targets

                )


                losses = sum(

                    loss

                    for loss in loss_dict.values()

                )



            scaler.scale(
                losses
            ).backward()



            scaler.step(
                optimizer
            )


            scaler.update()



        else:


            loss_dict = model(

                images,

                targets

            )


            losses = sum(

                loss

                for loss in loss_dict.values()

            )


            losses.backward()


            optimizer.step()



        total_loss += losses.item()



        progress.set_description(

            f"Loss: {losses.item():.4f}"

        )




    avg_loss = (

        total_loss /

        len(train_loader)

    )



    scheduler.step()



    print(

        "\nAverage Training Loss:",

        avg_loss

    )



    print(

        "Learning Rate:",

        optimizer.param_groups[0]["lr"]

    )



    # Save best model

    if avg_loss < best_loss:


        best_loss = avg_loss



        torch.save(

            model.state_dict(),

            MODEL_PATH

        )


        print(
            "Saved best model ✅"
        )




print("\n================================")

print(
    "Training Completed Successfully ✅"
)

print("================================")


print(

    "Best Loss:",

    best_loss

)