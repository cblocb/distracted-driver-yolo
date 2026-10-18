import argparse
import csv
import json
import os
import random
import shutil

DATASET_ROOT = "state-farm-distracted-driver-detection"
CSV_PATH = os.path.join(DATASET_ROOT, "driver_imgs_list.csv")
TRAIN_ROOT = os.path.join(DATASET_ROOT, "imgs", "train")
VAL_ROOT = "binary_dataset"
RUN_ROOT = "runs"
DEVICE = "cpu"
LINK_MODE = "copy"

SEED = 42
VAL_SUBJECTS = 5

MODEL_NAME = "yolo11n-cls.pt"
RUN_NAME = "level1_binary"
EPOCHS = 15
IMAGE_SIZE = 224
BATCH_SIZE = 32


def get_binary_label(class_name):
    if class_name not in {f"c{i}" for i in range(10)}:
        raise ValueError(f"Unknown source class: {class_name}")
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

    if not 0 < VAL_SUBJECTS < len(subjects):
        raise ValueError("Validation driver count must leave drivers in both partitions")
    random.Random(SEED).shuffle(subjects)
    return set(subjects[:VAL_SUBJECTS])


def make_split(rows, val_subjects):
    # Validate inputs before writing, and never remove an existing directory.
    copies = []
    destinations = set()
    for row in rows:
        image_name = row["img"]
        if not image_name or image_name in {".", ".."} or "/" in image_name or "\\" in image_name:
            raise ValueError("Image names must be basenames")
        label = get_binary_label(row["classname"])
        split = "val" if row["subject"] in val_subjects else "train"
        source = os.path.join(TRAIN_ROOT, row["classname"], image_name)
        target = os.path.join(VAL_ROOT, split, label, image_name)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        if target in destinations:
            raise ValueError(f"Duplicate output image: {image_name}")
        destinations.add(target)
        copies.append((source, target, split, label))

    os.makedirs(VAL_ROOT, exist_ok=False)

    os.makedirs(os.path.join(VAL_ROOT, "train", "focused"))
    os.makedirs(os.path.join(VAL_ROOT, "train", "distracted"))
    os.makedirs(os.path.join(VAL_ROOT, "val", "focused"))
    os.makedirs(os.path.join(VAL_ROOT, "val", "distracted"))

    summary = {
        "train": {"focused": 0, "distracted": 0},
        "val": {"focused": 0, "distracted": 0},
    }

    for source_path, target_path, split, binary_label in copies:
        if LINK_MODE == "symlink":
            os.symlink(os.path.abspath(source_path), target_path)
        else:
            shutil.copy2(source_path, target_path)
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
    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)

    model.train(
        data=VAL_ROOT,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=RUN_ROOT,
        name=RUN_NAME,
        device=DEVICE,
        seed=SEED,
        fliplr=0.5,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        scale=0.1,
        erasing=0.2,
        exist_ok=False,
    )

    metrics = model.val(
        data=VAL_ROOT,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        split="val",
    )

    metrics_path = os.path.join(RUN_ROOT, RUN_NAME, "val_metrics.txt")
    with open(metrics_path, "w") as file:
        file.write(str(metrics))

    print("Saved run to:", os.path.join(RUN_ROOT, RUN_NAME))


def main():
    global DATASET_ROOT, CSV_PATH, TRAIN_ROOT, VAL_ROOT, RUN_ROOT, DEVICE, LINK_MODE, RUN_NAME
    parser = argparse.ArgumentParser(description="Build a driver-disjoint split and optionally train YOLO11n")
    parser.add_argument("--dataset", required=True, help="Root containing driver_imgs_list.csv and imgs/train")
    parser.add_argument("--output", default="binary_dataset", help="New split directory; existing directories are refused")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--name", default="level1_binary")
    parser.add_argument("--device", default="cpu", help="Ultralytics device: cpu, mps or CUDA index such as 0")
    parser.add_argument("--link-mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare data without loading a model")
    args = parser.parse_args()
    DATASET_ROOT = os.path.abspath(args.dataset)
    CSV_PATH = os.path.join(DATASET_ROOT, "driver_imgs_list.csv")
    TRAIN_ROOT = os.path.join(DATASET_ROOT, "imgs", "train")
    VAL_ROOT, RUN_ROOT = os.path.abspath(args.output), os.path.abspath(args.runs)
    DEVICE, LINK_MODE, RUN_NAME = args.device, args.link_mode, args.name
    if not RUN_NAME or RUN_NAME in {".", ".."} or "/" in RUN_NAME or "\\" in RUN_NAME:
        parser.error("--name must be a simple directory name")
    if not args.prepare_only and os.path.exists(os.path.join(RUN_ROOT, RUN_NAME)):
        parser.error("Run directory already exists; choose a new --name")
    build_dataset()
    if not args.prepare_only:
        train_model()


if __name__ == "__main__":
    main()
