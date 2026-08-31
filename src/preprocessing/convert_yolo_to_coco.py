"""
Convert YOLO Detection Dataset -> COCO Format

Input:

data/multiclass/processed/

train/
    images/
    labels/

val/
    images/
    labels/

test/
    images/
    labels/


Output:

data/multiclass/annotations/

train.json
val.json
test.json

"""


from pathlib import Path
import json
import cv2
from tqdm import tqdm



# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(
    "data/multiclass/processed"
)


OUTPUT_DIR = Path(
    "data/multiclass/annotations"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# -----------------------------------
# Classes
# -----------------------------------

CLASSES = {

    0: "benign",
    1: "malignant",
    2: "ak",
    3: "bcc",
    4: "dermatofibroma",
    5: "melanoma",
    6: "nevus",
    7: "pigmented_bk",
    8: "scc",
    9: "vascular"

}



# -----------------------------------
# Conversion Function
# -----------------------------------

def convert_split(split):


    print("\nProcessing:", split)



    image_dir = (
        BASE_DIR /
        split /
        "images"
    )


    label_dir = (
        BASE_DIR /
        split /
        "labels"
    )



    output_file = (
        OUTPUT_DIR /
        f"{split}.json"
    )



    coco = {


        "images": [],


        "annotations": [],


        "categories": []

    }



    # -------------------------------
    # Categories
    # -------------------------------

    for class_id, name in CLASSES.items():


        coco["categories"].append(

            {

                "id": class_id,

                "name": name,

                "supercategory": "skin_lesion"

            }

        )




    annotation_id = 1



    images = list(
        image_dir.glob("*")
    )



    print(
        "Images found:",
        len(images)
    )



    # -------------------------------
    # Images
    # -------------------------------


    for image_id, img_path in enumerate(
        tqdm(
            images,
            desc=f"Converting {split}"
        )
    ):



        img = cv2.imread(
            str(img_path)
        )



        if img is None:

            print(
                "Skipping:",
                img_path.name
            )

            continue



        height, width = img.shape[:2]



        coco["images"].append(

            {

                "id": image_id,

                "file_name":
                    img_path.name,

                "width":
                    width,

                "height":
                    height

            }

        )



        label_file = (

            label_dir /
            f"{img_path.stem}.txt"

        )



        if not label_file.exists():

            print(
                "Missing label:",
                img_path.name
            )

            continue



        with open(
            label_file,
            "r"
        ) as f:

            lines = f.readlines()




        for line in lines:



            values = (
                line.strip()
                .split()
            )



            if len(values) != 5:

                continue




            class_id = int(
                values[0]
            )


            x_center = float(
                values[1]
            )


            y_center = float(
                values[2]
            )


            box_width = float(
                values[3]
            )


            box_height = float(
                values[4]
            )



            # YOLO -> COCO


            x = (

                x_center -
                box_width / 2

            ) * width



            y = (

                y_center -
                box_height / 2

            ) * height



            w = (

                box_width *
                width

            )



            h = (

                box_height *
                height

            )




            coco["annotations"].append(

                {

                    "id":
                        annotation_id,


                    "image_id":
                        image_id,


                    "category_id":
                        class_id,


                    "bbox":

                    [

                        x,
                        y,
                        w,
                        h

                    ],


                    "area":
                        w*h,


                    "iscrowd":
                        0

                }

            )


            annotation_id += 1





    # -------------------------------
    # Save JSON
    # -------------------------------


    with open(
        output_file,
        "w"
    ) as f:


        json.dump(

            coco,

            f,

            indent=4

        )



    print(
        "\nSaved:",
        output_file
    )

    print(
        "Images:",
        len(coco["images"])
    )

    print(
        "Annotations:",
        len(coco["annotations"])
    )





# -----------------------------------
# Run
# -----------------------------------


for split in [

    "train",
    "val",
    "test"

]:

    convert_split(split)



print(
    "\nCOCO Conversion Completed Successfully ✅"
)