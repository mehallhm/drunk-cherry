import keras.src.layers as layers
import keras.models as models
from keras.src.optimizers import Adam
from keras.src.losses import CategoricalCrossentropy

from trail_helpers import DIFFICULTY_MAP

NUM_CLASSES = len(set(DIFFICULTY_MAP.values()))
NUM_OBSERVATIONS = 100


def model(pts: int):
    """
    Build the MLP model

    :param pts: number of input points
    :return:
    """

    mlp = models.Sequential(
        [
            layers.Input(shape=(pts, 3)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(NUM_CLASSES - 1, activation="softmax"),
        ]
    )

    mlp.summary()

    mlp.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])

    return mlp
