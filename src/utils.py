from pathlib import Path

import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

seed = 1337

difficulty_mapping = {
    "Easy": "Easy",
    "Easy_Intermediate": "Easy",
    "Intermediate": "Intermediate",
    "Intermediate_Difficult": "Intermediate_Difficult",
    "Difficult": "Difficult",
    "Very_Difficult": "Difficult",
}


def parse_stem(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format {stem}")
    trail_id = parts[0]
    difficulty = "_".join(parts[2:])
    return trail_id, difficulty


def parse_difficulty(stem: str) -> str:
    _, difficulty = parse_stem(stem)
    return difficulty


def parse_trail_id(stem: str) -> str:
    trail_id, _ = parse_stem(stem)
    return trail_id


def get_dataset_info(path: Path) -> dict:
    png_files = collect_pngs(path)

    with Image.open(png_files[0]) as img:
        img_width, img_height = img.size

    class_counts: dict[str, int] = {}
    trail_ids: set[str] = set()

    for f in png_files:
        trail_id, difficulty = parse_stem(f.stem)
        difficulty = combine_difficulties(difficulty)
        trail_ids.add(trail_id)
        class_counts[difficulty] = class_counts.get(difficulty, 0) + 1

    total = len(png_files)
    classes = sorted(class_counts.keys())

    return {
        "total": total,
        "img_size": (img_height, img_width),
        "classes": classes,
        "num_classes": len(classes),
        "class_counts": class_counts,
        "class_ratio": {k: v / total for k, v in class_counts.items()},
        "trail_ids": trail_ids,
        "min_class_count": min(class_counts.values()),
        "max_class_count": max(class_counts.values()),
    }


def combine_difficulties(difficulty: str) -> str:
    if difficulty not in difficulty_mapping:
        raise ValueError(f"Unknown difficulty label: '{difficulty}'")
    return difficulty_mapping[difficulty]


def import_data(
    path: Path,
    balance: str = "none",
    test_size: float = 0.25,
) -> tuple:
    png_files = collect_pngs(path)

    with Image.open(png_files[0]) as img:
        img_width, img_height = img.size
    img_size = (img_height, img_width)

    images_by_class = {}
    for f in png_files:
        difficulty = combine_difficulties(parse_difficulty(f.stem))
        images_by_class.setdefault(difficulty, []).append(f)

    if balance == "undersample":
        min_count = min(len(v) for v in images_by_class.values())
        rng = np.random.default_rng(seed)
        selected: list[Path] = []
        for files in images_by_class.values():
            chosen = rng.choice(files, size=min_count, replace=False)  # type: ignore[arg-type]
            selected.extend(chosen.tolist())
        png_files = selected

        print(f"min_count: {min_count}")
    else:
        png_files = list(png_files)  # already a list; make a copy

    # X
    # TODO: make sure to extract the max grade, elevation gain, and elevation loss from the dataset here too.
    # We'll need it for the model fitting now since we want to add some features after CNN flatten opration
    images = []
    # y
    labels = []

    for f in png_files:
        difficulty = combine_difficulties(parse_difficulty(f.stem))
        with Image.open(f) as img:
            arr = np.array(img.convert("L"), dtype=np.float32)
        images.append(arr)
        labels.append(difficulty)

    X = np.stack(images, axis=0)[..., np.newaxis]  # (N, H, W, 1)

    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    num_classes = len(le.classes_)
    y_onehot = keras.utils.to_categorical(y_int, num_classes=num_classes)

    # Print class distribution after optional balancing
    unique, counts = np.unique(y_int, return_counts=True)
    print(f"{num_classes} classes (balance='{balance}'):")
    for idx, cnt in zip(unique, counts):
        print(f"  {le.classes_[idx]}: {cnt / len(labels):.1%}  ({cnt} images)")

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y_onehot, test_size=test_size, random_state=seed, stratify=y_int
    )

    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    return (xTrain, xTest, yTrain, yTest, le, img_size)


def collect_pngs(path: Path) -> list[Path]:
    path = Path(path)
    png_files = sorted(
        f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".png"
    )
    if not png_files:
        raise FileNotFoundError(f"No PNG images found at {path}")
    return png_files
