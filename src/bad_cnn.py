import numpy as np

# import pandas as pd
import argparse
from pathlib import Path
# import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout

# from keras.optimizers import Adam
# from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from PIL import Image

seed = 1337
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

# multi class classification
def create_model(img_size: tuple[int, int], num_classes: int) -> Model:
    height, width = img_size
    inpx = Input(shape=(height, width, 1))

    x = Conv2D(1, (3, 3), padding="same", activation="relu")(inpx)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(2, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(4, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D((3, 3))(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)

    out_layer = Dense(num_classes, activation="softmax")(x)

    model = Model([inpx], out_layer)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def parse_difficulty(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format {stem}")
    return "_".join(parts[2:])


def train_test_model(
    model: Model,
    data: tuple,
    epochs: int,
    output_path: Path = Path("trail_cnn_model.keras"),
) -> None:
    xTrain, xTest, yTrain, yTest = data

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        xTrain,
        yTrain,
        epochs=epochs,
        validation_data=(xTest, yTest),
        callbacks=[early_stop],
    )

    score = model.evaluate(xTest, yTest)
    print("\nloss = ", score[0])
    print("accuracy = ", score[1])

    model.save(output_path)
    print(f"Model saved to {output_path}")


# world's most sloppily written function but it's alright...
def import_data(path: Path) -> tuple:
    png_files = [
        f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".png"
    ]
    if not png_files:
        raise FileNotFoundError(f"No PNG images found at {path}")

    with Image.open(png_files[0]) as img:
        img_width, img_height = img.size
    img_size = (img_height, img_width)
    print(f"Image sizes are {img_height} by {img_width}")

    # X
    images = []
    # y
    labels = []

    label_counts = {}

    for f in png_files:
        difficulty = parse_difficulty(f.stem)

        with Image.open(f) as img:
            arr = np.array(img.convert("L"), dtype=np.float32)

        images.append(arr)
        labels.append(difficulty)
        label_counts[difficulty] = label_counts.get(difficulty, 0) + 1

    X = np.stack(images, axis=0)[..., np.newaxis]

    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    num_classes = len(le.classes_)

    print(f"{num_classes} classes with following distributions:")
    for label, count in label_counts.items():
        print(f"{label}: {count/len(images)}")

    y_onehot = keras.utils.to_categorical(y_int, num_classes=num_classes)

    xTrain, xTest, yTrain, yTest = train_test_split(
        X, y_onehot, test_size=0.25, random_state=seed, stratify=y_int
    )

    xTrain = xTrain / 255.0
    xTest = xTest / 255.0
    return (xTrain, xTest, yTrain, yTest, le, img_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="initial CNN model for classification of bike GPS data to difficulty"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="IMG_DATA_PATH",
        help="Path to the input image data directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="N",
        help="Number of epochs to train the CNN over",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("trail_cnn_model.keras"),
        metavar="MODEL_PATH",
        help="Where to save trained model",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    path: Path = args.input
    epochs: int = args.epochs
    output: Path = args.output

    if not path.is_dir():
        parser.error(f"Input directory {path} is not a valid directory")

    xTrain, xTest, yTrain, yTest, label_encoder, img_size = import_data(path)
    num_classes = len(label_encoder.classes_)

    model = create_model(img_size, num_classes)
    model.summary()

    train_test_model(model, (xTrain, xTest, yTrain, yTest), epochs, output)

    print("done")


if __name__ == "__main__":
    main()
