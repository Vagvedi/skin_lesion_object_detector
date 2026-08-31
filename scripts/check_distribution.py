import json
from collections import Counter


data = json.load(
    open(
        "data/multiclass/annotations/train.json"
    )
)


counter = Counter()


for ann in data["annotations"]:
    counter[ann["category_id"]] += 1



classes = [
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


for i,c in enumerate(classes):

    print(
        c,
        ":",
        counter[i]
    )