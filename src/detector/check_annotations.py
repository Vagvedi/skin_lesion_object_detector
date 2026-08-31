import json
import cv2
import random
import os


ANN_FILE = "data/detection/annotations/test.json"

IMAGE_DIR = "data/detection/test/images"


with open(ANN_FILE) as f:
    coco = json.load(f)


image = random.choice(coco["images"])

print("Testing:", image["file_name"])


img_path = os.path.join(
    IMAGE_DIR,
    image["file_name"]
)


img = cv2.imread(img_path)


for ann in coco["annotations"]:

    if ann["image_id"] == image["id"]:

        x,y,w,h = ann["bbox"]


        cv2.rectangle(
            img,
            (int(x),int(y)),
            (int(x+w),int(y+h)),
            (0,255,0),
            3
        )


        print(
            "GT BOX:",
            ann["bbox"]
        )


cv2.imshow(
    "Ground Truth",
    img
)

cv2.waitKey(0)