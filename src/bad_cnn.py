import numpy as np

# import pandas as pd
import argparse
from pathlib import Path
# import matplotlib.pyplot as plt

import keras
from keras.models import Model
from keras.layers import (
    Dense,
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    LeakyReLU,
    Concatenate,
)

# from keras.optimizers import Adam
# from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from PIL import Image
import matplotlib.pyplot as plt

import utils

seed = 1337
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)


def save_feature_map(arr: np.ndarray, path: Path, cmap: str = "viridis") -> None:
    a_min, a_max = arr.min(), arr.max()
    norm = (arr - a_min) / (a_max - a_min) if a_max - a_min > 0 else np.zeros_like(arr)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(norm, cmap=cmap, interpolation="nearest")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=100)
    plt.close(fig)


def create_kernel_images(
    model: Model, inputs: tuple, output_dir: Path = Path("./visuals/kernel_plots/")
):
    xTest, fTest = inputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    sample_idx = int(rng.integers(0, len(xTest)))
    x_sample = xTest[sample_idx]
    f_sample = fTest[sample_idx]
    sample_input_x = x_sample[np.newaxis]
    sample_input_f = f_sample[np.newaxis]

    save_feature_map(x_sample[..., 0], output_dir / "raw_sample.png", cmap="gray")

    for layer in model.layers:
        if not isinstance(layer, Conv2D):
            continue

        num_filters = layer.get_weights()[0].shape[-1]
        activation_model = Model(inputs=model.input, outputs=layer.output)
        activations = activation_model.predict((sample_input_x, sample_input_f))

        for filter_idx in range(num_filters):
            act_map = activations[0, :, :, filter_idx]
            save_feature_map(
                act_map,
                output_dir / f"{layer.name}_filter_{filter_idx}_activation.png",
            )


# multi class classification
def create_model(img_size: tuple[int, int], num_classes: int) -> Model:
    height, width = img_size
    input_image = Input(shape=(height, width, 1))
    # TODO: fix hardcoded values
    # adds some extra features after the convolution layers
    extra_features = Input(shape=(4,))

    # x = Conv2D(1, (3, 3), padding="same", activation="relu")(input_image)
    # x = MaxPooling2D((2, 2))(x)
    #
    # x = Conv2D(2, (3, 3), padding="valid", activation="relu")(x)
    # x = MaxPooling2D((2, 2))(x)
    #
    # x = Conv2D(4, (3, 3), padding="valid", activation="relu")(x)
    # x = MaxPooling2D((2, 2))(x)
    #
    # x = Conv2D(8, (3, 3), padding="valid", activation="relu")(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(16, (3, 3), padding="valid", activation="relu")(x)
    # x = MaxPooling2D((2, 2))(x)

    # x = Conv2D(32, (3, 3), padding="valid", activation="relu")(x)
    # x = MaxPooling2D((3, 3))(x)

    post_convolution = Flatten()(input_image)
    x = Concatenate()([post_convolution, extra_features])

    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)

    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)

    x = Dense(32)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.7)(x)

    out_layer = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=[input_image, extra_features], outputs=out_layer)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def train_test_model(
    model: Model,
    data: tuple,
    epochs: int,
    output_path: Path = Path("trail_cnn_model.keras"),
) -> None:
    xTrain, xTest, fTrain, fTest, yTrain, yTest = data

    early_stop = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        [xTrain, fTrain],
        yTrain,
        epochs=epochs,
        validation_data=((xTest, fTest), yTest),
        callbacks=[early_stop],
    )

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
        default="none",
        metavar="BALANCE",
        help="Undersampling: Clip all classes to the smallest",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    path: Path = args.input
    epochs: int = args.epochs
    output: Path = args.output
    balance = args.balance

    if not path.is_dir():
        parser.error(f"Input directory {path} is not a valid directory")

    # print(utils.get_dataset_info(path))

    # NOTE: I know it's a very not good idea to have the hardcoded csv. This is for testing
    xTrain, xTest, yTrain, yTest, fTrain, fTest, label_encoder, img_size = (
        utils.import_data(path, Path("../all_trails.csv"), balance=balance)
    )
    num_classes = len(label_encoder.classes_)

    model = create_model(img_size, num_classes)
    model.summary()

    train_test_model(
        model, (xTrain, xTest, fTrain, fTest, yTrain, yTest), epochs, output
    )

    create_kernel_images(model, (xTest, fTest))

    print("done")


if __name__ == "__main__":
    main()
