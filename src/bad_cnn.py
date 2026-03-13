import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

IMG_SIZE = 512


# ripped this architecture from my homework 2. Will have to tweak this
def create_model():
    inpx = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = Conv2D(1, (3, 3), padding="same", activation="tanh")(inpx)

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

    out_layer = Dense(1, activation="sigmoid")(x)

    model = Model([inpx], out_layer)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


# do the training...
def train_test_model(model, data, epochs=20):
    xTrain, yTrain, xTest, yTest = data

    model.fit(xTrain, yTrain, epochs=epochs, validation_data=(xTest, yTest))
    score = model.evaluate(xTest, yTest, verbose=0)
    print("loss = ", score[0])
    print("accuracy = ", score[1])
    pass


def import_data(path: Path) -> np.ndarray:
    for f in path.iterdir():
        if f.is_file(follow_symlinks=False) and f.suffix == ".png":
            # TODO: import the image, parse the file name for difficulty and trail id
            pass

    # TODO: train test split the data we've read
    # xTrain, xTest, yTrain, yTest = train_test_split(
    #     data, y_categories, test_size=0.25, random_state=seed
    # )
    # xTrain /= 255.0
    # xTest /= 255.0
    return np.array(())


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

    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()
    path: Path = args.input
    epochs = args.epochs

    if not path.is_dir():
        parser.error(f"Input directory {path} is not a valid directory")

    # (xTrain, yTrain, xTest, yTest)
    data = import_data(path)

    model = create_model()
    train_test_model(model, data, epochs)

    print("done")


if __name__ == "__main__":
    main()
