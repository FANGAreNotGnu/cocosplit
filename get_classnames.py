import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np


parser = argparse.ArgumentParser(
    description="Splits COCO annotations file into training and test sets."
)
parser.add_argument(
    "annotations",
    metavar="coco_annotations",
    type=str,
    help="Path to COCO annotations file.",
)
args = parser.parse_args()


def main(args):
    annotation_file = args.annotations
    with open(annotation_file, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        images = coco["images"]
        print(len(images))
        annotations = coco["annotations"]
        print(len(annotations))
        categories = coco["categories"]
        cat_names = [c["name"] for c in categories]
        print(cat_names)

if __name__ == "__main__":
    main(args)
