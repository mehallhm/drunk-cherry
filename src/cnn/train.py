import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import argparse
from pathlib import Path

import keras
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

import utils
from cnn.model import create_model
from cnn.utils import create_kernel_images

seed = 1337
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)


def train_test_model(
    model,
    data: tuple,
    epochs: int,
    output_path: Path = Path("trail_cnn_model.keras"),
) -> None:
    xTrain, xTest, fTrain, fTest, yTrain, yTest = data

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    history = model.fit(
        [xTrain, fTrain],
        yTrain,
        epochs=epochs,
        validation_data=((xTest, fTest), yTest),
        # callbacks=[early_stop],
    )

    plt.figure()
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"])
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("loss_plots.png")

    score = model.evaluate(
        [xTest, fTest],
        yTest,
    )
    print("\nloss = ", score[0])
    print("accuracy = ", score[1])

    model.save(output_path)
    print(f"Model saved to {output_path}")


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

    parser.add_argument(
        "--balance",
        type=str,
        choices=["undersample", "none"],
        default="undersample",
        metavar="BALANCE",
        help="Undersampling: Clip all classes to the smallest",
    )

    parser.add_argument(
        "--color",
        type=str,
        choices=["rgb", "gray"],
        default="gray",
        metavar="BALANCE",
        help="Whether to use RGB color channels or grayscale color channels",
    )

    parser.add_argument(
        "--csv_path",
        type=Path,
        default="../all_trails.csv",
        help="Where the csv for all trail data is located",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    path: Path = args.input
    epochs: int = args.epochs
    output: Path = args.output
    balance = args.balance
    color = args.color
    csv_path: Path = args.csv_path

    if not path.is_dir():
        parser.error(f"Input directory {path} is not a valid directory")

    if not csv_path.is_file() or csv_path.suffix == "csv":
        parser.error(f"CSV path {path} is not a valid csv file")

    # TODO: Should be able to pull this information from the first image in the dataset instead of requiring a argument...
    channels = 1 if color == "gray" else 3

    xTrain, xTest, yTrain, yTest, fTrain, fTest, label_encoder, img_size = (
        utils.import_data(path, csv_path, balance=balance, channels=channels)
    )

    num_classes = len(label_encoder.classes_)

    model = create_model(img_size, num_classes, len(fTrain[0]))
    model.summary()

    train_test_model(
        model, (xTrain, xTest, fTrain, fTest, yTrain, yTest), epochs, output
    )

    create_kernel_images(model, (xTest, fTest))

    print("done")


if __name__ == "__main__":
    main()
