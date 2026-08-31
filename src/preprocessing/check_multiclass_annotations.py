from pathlib import Path
import cv2
import random


IMAGE_DIR = Path(
    "data/multiclass/processed/train/images"
)

LABEL_DIR = Path(
    "data/multiclass/processed/train/labels"
)


images = list(IMAGE_DIR.glob("*"))

img_path = random.choice(images)

label_path = (
    LABEL_DIR /
    f"{img_path.stem}.txt"
)


print("Testing:", img_path.name)


img = cv2.imread(str(img_path))

h,w,_ = img.shape


with open(label_path) as f:
    lines = f.readlines()


for line in lines:

    cls,x,y,bw,bh = map(
        float,
        line.split()
    )


    x1 = int((x-bw/2)*w)
    y1 = int((y-bh/2)*h)

    x2 = int((x+bw/2)*w)
    y2 = int((y+bh/2)*h)


    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        3
    )


    cv2.putText(
        img,
        f"class {int(cls)}",
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )


cv2.imshow(
    "Annotation Check",
    img
)

cv2.waitKey(0)
cv2.destroyAllWindows()