import os

from ultralytics import YOLO


VAL_ROOT = "/Users/caleb/git/toys/cv/project2/binary_dataset/val"
RUN_ROOT = "/Users/caleb/git/toys/cv/project2/runs/level1_binary"
MODEL_PATH = os.path.join(RUN_ROOT, "weights", "best.pt")
OUT_PATH = os.path.join(RUN_ROOT, "validation_metrics_summary.txt")

IMAGE_SIZE = 224
BATCH_SIZE = 32


def load_val_images():
    image_paths = []
    true_labels = []

    for label in ["focused", "distracted"]:
        label_dir = os.path.join(VAL_ROOT, label)
        for image_name in sorted(os.listdir(label_dir)):
            image_paths.append(os.path.join(label_dir, image_name))
            true_labels.append(label)

    return image_paths, true_labels


def get_metrics(true_labels, pred_labels):
    matrix = {
        "focused": {"focused": 0, "distracted": 0},
        "distracted": {"focused": 0, "distracted": 0},
    }

    for true_label, pred_label in zip(true_labels, pred_labels):
        matrix[true_label][pred_label] += 1

    results = {}
    for label in ["focused", "distracted"]:
        tp = matrix[label][label]
        fp = 0
        fn = 0

        for other_label in ["focused", "distracted"]:
            if other_label != label:
                fp += matrix[other_label][label]
                fn += matrix[label][other_label]

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        support = matrix[label]["focused"] + matrix[label]["distracted"]

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    correct = matrix["focused"]["focused"] + matrix["distracted"]["distracted"]
    accuracy = correct / len(true_labels)
    majority_baseline = max(results["focused"]["support"], results["distracted"]["support"]) / len(true_labels)

    weighted_f1 = (
        results["focused"]["f1"] * results["focused"]["support"]
        + results["distracted"]["f1"] * results["distracted"]["support"]
    ) / len(true_labels)

    macro_f1 = (results["focused"]["f1"] + results["distracted"]["f1"]) / 2

    return matrix, results, accuracy, majority_baseline, weighted_f1, macro_f1


def main():
    image_paths, true_labels = load_val_images()

    model = YOLO(MODEL_PATH)
    predictions = model.predict(
        source=image_paths,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device="mps",
        verbose=False,
    )

    pred_labels = []
    for prediction in predictions:
        pred_index = int(prediction.probs.top1)
        pred_labels.append(prediction.names[pred_index])

    matrix, results, accuracy, majority_baseline, weighted_f1, macro_f1 = get_metrics(true_labels, pred_labels)

    lines = []
    lines.append(f"Accuracy: {accuracy:.6f}")
    lines.append(f"Majority class baseline accuracy: {majority_baseline:.6f}")

    for label in ["focused", "distracted"]:
        lines.append(
            f"{label}: precision={results[label]['precision']:.6f} "
            f"recall={results[label]['recall']:.6f} "
            f"f1={results[label]['f1']:.6f} "
            f"support={results[label]['support']}"
        )

    lines.append(f"Macro F1: {macro_f1:.6f}")
    lines.append(f"Weighted F1: {weighted_f1:.6f}")
    lines.append("Confusion matrix (rows=true, cols=pred) [focused, distracted]:")
    lines.append(f"{matrix['focused']['focused']}, {matrix['focused']['distracted']}")
    lines.append(f"{matrix['distracted']['focused']}, {matrix['distracted']['distracted']}")

    with open(OUT_PATH, "w") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
