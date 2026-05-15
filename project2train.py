import csv
import json
import os
import random
import shutil

from ultralytics import YOLO


DATASET_ROOT = "/Users/caleb/git/toys/cv/project2/state-farm-distracted-driver-detection"
CSV_PATH = os.path.join(DATASET_ROOT, "driver_imgs_list.csv")
TRAIN_ROOT = os.path.join(DATASET_ROOT, "imgs", "train")
VAL_ROOT = "/Users/caleb/git/toys/cv/project2/binary_dataset"
RUN_ROOT = "/Users/caleb/git/toys/cv/project2/runs"

SEED = 42
VAL_SUBJECTS = 5

MODEL_NAME = "yolo11n-cls.pt"
RUN_NAME = "level1_binary"
EPOCHS = 15
IMAGE_SIZE = 224
BATCH_SIZE = 32


def get_binary_label(class_name):
    if class_name == "c0":
        return "focused"
    return "distracted"


def load_rows():
    rows = []
    with open(CSV_PATH, newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def choose_val_subjects(rows):
    subjects = []
    for row in rows:
        subject = row["subject"]
        if subject not in subjects:
            subjects.append(subject)

    random.seed(SEED)
    random.shuffle(subjects)
    return set(subjects[:VAL_SUBJECTS])


def make_split(rows, val_subjects):
    if os.path.exists(VAL_ROOT):
        shutil.rmtree(VAL_ROOT)

    os.makedirs(os.path.join(VAL_ROOT, "train", "focused"))
    os.makedirs(os.path.join(VAL_ROOT, "train", "distracted"))
    os.makedirs(os.path.join(VAL_ROOT, "val", "focused"))
    os.makedirs(os.path.join(VAL_ROOT, "val", "distracted"))

    summary = {
        "train": {"focused": 0, "distracted": 0},
        "val": {"focused": 0, "distracted": 0},
    }

    for row in rows:
        subject = row["subject"]
        class_name = row["classname"]
        image_name = row["img"]

        split = "val"
        if subject not in val_subjects:
            split = "train"

        binary_label = get_binary_label(class_name)
        source_path = os.path.join(TRAIN_ROOT, class_name, image_name)
        target_path = os.path.join(VAL_ROOT, split, binary_label, image_name)

        os.symlink(os.path.abspath(source_path), target_path)
        summary[split][binary_label] += 1

    return summary


def save_split_summary(rows, val_subjects, summary):
    class_counts = {}
    for row in rows:
        class_name = row["classname"]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

    summary_data = {
        "val_subjects": sorted(list(val_subjects)),
        "num_images": len(rows),
        "original_class_counts": class_counts,
        "binary_split_counts": summary,
    }

    out_path = os.path.join(VAL_ROOT, "split_summary.json")
    with open(out_path, "w") as file:
        json.dump(summary_data, file, indent=2)


def build_dataset():
    rows = load_rows()
    val_subjects = choose_val_subjects(rows)
    summary = make_split(rows, val_subjects)
    save_split_summary(rows, val_subjects, summary)

    print("Built binary dataset at:", VAL_ROOT)
    print("Validation subjects:", sorted(list(val_subjects)))
    print("Train counts:", summary["train"])
    print("Val counts:", summary["val"])


def train_model():
    model = YOLO(MODEL_NAME)

    model.train(
        data=VAL_ROOT,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=RUN_ROOT,
        name=RUN_NAME,
        device="mps",
        seed=SEED,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        scale=0.1,
        erasing=0.2,
        exist_ok=True,
    )

    metrics = model.val(
        data=VAL_ROOT,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="mps",
        split="val",
    )

    metrics_path = os.path.join(RUN_ROOT, RUN_NAME, "val_metrics.txt")
    with open(metrics_path, "w") as file:
        file.write(str(metrics))

    print("Saved run to:", os.path.join(RUN_ROOT, RUN_NAME))


def main():
    build_dataset()
    train_model()


if __name__ == "__main__":
    main()
