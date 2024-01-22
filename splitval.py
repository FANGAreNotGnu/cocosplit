import json
import argparse
import funcy
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
import numpy as np


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, "wt", encoding="UTF-8") as coco:
        json.dump(
            {
                "info": info,
                "licenses": licenses,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            coco,
            indent=2,
            sort_keys=True,
        )


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    return funcy.lfilter(lambda a: int(a["image_id"]) in image_ids, annotations)


def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i["image_id"]), annotations)

    return funcy.lfilter(lambda a: int(a["id"]) in annotation_ids, images)


parser = argparse.ArgumentParser(
    description="Splits COCO annotations file into training and test sets."
)
parser.add_argument(
    "annotations",
    metavar="coco_annotations",
    type=str,
    help="Path to COCO annotations file.",
)
parser.add_argument(
    "-a",
    "--having-annotations",
    dest="having_annotations",
    action="store_true",
    help="Ignore all images without annotations. Keep only these with at least one annotation",
)

parser.add_argument(
    "-m",
    "--multi-class",
    dest="multi_class",
    action="store_true",
    help="Split a multi-class dataset while preserving class distributions in train and test sets",
)

args = parser.parse_args()


# implementation in Autogluon
def holdout_frac(num_train_rows):
    if num_train_rows < 5000:
        holdout_frac = max(0.1, min(0.2, 500.0 / num_train_rows))
    else:
        holdout_frac = max(0.01, min(0.1, 2500.0 / num_train_rows))

    return holdout_frac


def main(args):
    annotation_file = args.annotations
    train_file = annotation_file[:-5] + "_train.json"
    val_file = annotation_file[:-5] + "_val.json"

    with open(annotation_file, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)
        info = coco["info"] if "info" in coco else ""
        licenses = coco["licenses"] if "info" in coco else ""
        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        images_with_annotations = funcy.lmap(lambda a: int(a["image_id"]), annotations)

        if args.having_annotations:
            images = funcy.lremove(
                lambda i: i["id"] not in images_with_annotations, images
            )

        split = 1 - holdout_frac(len(images))

        if args.multi_class:
            annotation_categories = funcy.lmap(
                lambda a: int(a["category_id"]), annotations
            )

            # bottle neck 1
            # remove classes that has only one sample, because it can't be split into the training and testing sets
            annotation_categories = funcy.lremove(
                lambda i: annotation_categories.count(i) <= 1, annotation_categories
            )

            annotations = funcy.lremove(
                lambda i: i["category_id"] not in annotation_categories, annotations
            )

            X_train, y_train, X_test, y_test = iterative_train_test_split(
                np.array([annotations]).T,
                np.array([annotation_categories]).T,
                test_size=1 - split,
            )

            save_coco(
                train_file,
                info,
                licenses,
                filter_images(images, X_train.reshape(-1)),
                X_train.reshape(-1).tolist(),
                categories,
            )
            save_coco(
                val_file,
                info,
                licenses,
                filter_images(images, X_test.reshape(-1)),
                X_test.reshape(-1).tolist(),
                categories,
            )

            print(
                "Saved {} entries in {} and {} in {}".format(
                    len(X_train), train_file, len(X_test), val_file
                )
            )

        else:
            X_train, X_test = train_test_split(images, train_size=split)

            anns_train = filter_annotations(annotations, X_train)
            anns_test = filter_annotations(annotations, X_test)

            save_coco(train_file, info, licenses, X_train, anns_train, categories)
            save_coco(val_file, info, licenses, X_test, anns_test, categories)

            print(
                "Saved {} entries in {} and {} in {}".format(
                    len(anns_train), train_file, len(anns_test), val_file
                )
            )


if __name__ == "__main__":
    main(args)
