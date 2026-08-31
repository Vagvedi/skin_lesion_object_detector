"""
Merge 10 skin lesion YOLO datasets into one multi-class dataset.

Input:
data/raw/
    benign/
    malignant/
    ak/
    bcc/
    dermatofibroma/
    melanoma/
    nevus/
    pigmented bk/
    scc/
    vascular/

Output:
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
"""


from pathlib import Path
import shutil
import random



# ==============================
# Paths
# ==============================

RAW_DIR = Path(
    "data/raw"
)


OUTPUT_DIR = Path(
    "data/multiclass/processed"
)



# ==============================
# Class Mapping
# ==============================

CLASS_MAPPING = {

    "benign": 0,
    "malignant": 1,

    "ak": 2,
    "bcc": 3,
    "dermatofibroma": 4,
    "melanoma": 5,
    "nevus": 6,
    "pigmented bk": 7,
    "scc": 8,
    "vascular": 9

}


CLASS_NAMES = [

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



# ==============================
# Split Ratio
# ==============================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2



# ==============================
# Create folders
# ==============================

for split in [
    "train",
    "val",
    "test"
]:

    (OUTPUT_DIR / split / "images").mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_DIR / split / "labels").mkdir(
        parents=True,
        exist_ok=True
    )



# ==============================
# Split Function
# ==============================

def get_split():

    r = random.random()


    if r < TRAIN_RATIO:
        return "train"

    elif r < TRAIN_RATIO + VAL_RATIO:
        return "val"

    else:
        return "test"



# ==============================
# Processing
# ==============================

random.seed(42)


total_images = 0



for class_name, class_id in CLASS_MAPPING.items():


    print(
        "\nProcessing:",
        class_name
    )


    class_folder = (
        RAW_DIR /
        class_name
    )


    image_folder = (
        class_folder /
        "images"
    )


    label_folder = (
        class_folder /
        "labels" /
        "labels" /
        "Train"
    )


    if not image_folder.exists():

        print(
            "Missing image folder:",
            class_name
        )

        continue



    images = list(
        image_folder.glob("*")
    )


    print(
        "Images found:",
        len(images)
    )



    for image_path in images:


        label_path = (
            label_folder /
            f"{image_path.stem}.txt"
        )



        if not label_path.exists():

            print(
                "Missing label:",
                image_path.name
            )

            continue



        split = get_split()



        # Avoid duplicate ISIC names

        new_filename = (
            f"{class_name.replace(' ','_')}_"
            f"{image_path.name}"
        )


        new_image_path = (
            OUTPUT_DIR /
            split /
            "images" /
            new_filename
        )


        new_label_path = (
            OUTPUT_DIR /
            split /
            "labels" /
            f"{Path(new_filename).stem}.txt"
        )



        shutil.copy(

            image_path,

            new_image_path

        )



        # ==============================
        # Rewrite YOLO labels
        # ==============================

        new_lines = []


        with open(
            label_path,
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



            # Replace old dataset ID
            # with global ID

            values[0] = str(
                class_id
            )


            new_lines.append(
                " ".join(values)
            )



        with open(
            new_label_path,
            "w"
        ) as f:

            f.write(
                "\n".join(new_lines)
            )



        total_images += 1



print("\n==============================")
print("Dataset Merge Completed ✅")
print("==============================")

print(
    "Total images:",
    total_images
)



# ==============================
# data.yaml
# ==============================


yaml_content = f"""
path: {OUTPUT_DIR}

train: train/images
val: val/images
test: test/images

names:
"""


for idx,name in enumerate(CLASS_NAMES):

    yaml_content += (
        f"  {idx}: {name}\n"
    )



with open(
    OUTPUT_DIR / "data.yaml",
    "w"
) as f:

    f.write(
        yaml_content
    )


print(
    "\ndata.yaml created ✅"
)