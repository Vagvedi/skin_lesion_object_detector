import os
import json

from PIL import Image

import torch
from torch.utils.data import Dataset



class SkinLesionDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        transforms=None
    ):

        self.image_dir = image_dir
        self.transforms = transforms


        with open(annotation_file) as f:
            self.coco = json.load(f)


        self.images = self.coco["images"]


        self.annotations = {}


        for ann in self.coco["annotations"]:

            image_id = ann["image_id"]


            if image_id not in self.annotations:

                self.annotations[image_id] = []


            self.annotations[image_id].append(ann)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, idx):


        image_info = self.images[idx]


        image_path = os.path.join(
            self.image_dir,
            image_info["file_name"]
        )


        image = Image.open(
            image_path
        ).convert("RGB")



        anns = self.annotations.get(
            image_info["id"],
            []
        )


        boxes = []
        labels = []



        for ann in anns:


            x,y,w,h = ann["bbox"]


            boxes.append(

                [
                    x,
                    y,
                    x+w,
                    y+h
                ]

            )


            # COCO id + background offset

            labels.append(
                ann["category_id"] + 1
            )



        if len(boxes)==0:


            boxes = torch.zeros(
                (0,4),
                dtype=torch.float32
            )


            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )


        else:


            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )


            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )



        target = {


            "boxes": boxes,


            "labels": labels,


            # FIXED IMAGE ID

            "image_id": torch.tensor(
                image_info["id"]
            ),



            "area": torch.tensor(

                [
                    (b[2]-b[0]) *
                    (b[3]-b[1])

                    for b in boxes

                ],

                dtype=torch.float32

            ),



            "iscrowd": torch.zeros(

                len(boxes),

                dtype=torch.int64

            )

        }



        if self.transforms:

            image = self.transforms(image)



        return image,target