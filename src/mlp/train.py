import os

os.environ["KERAS_BACKEND"] = "torch"

import argparse
from pathlib import Path

import pandas as pd
from keras import backend as K

from trail_helpers import prepare_trail_data
from mlp.model import model, NUM_OBSERVATIONS
from mlp.utils import df_to_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MLP model for classification of bike GPS data to difficulty"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        metavar="DATA_PATH",
        help="Path to the trail data zip file",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="Number of epochs to train over",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="Training batch size",
    )

    parser.add_argument(
        "--points",
        type=int,
        default=NUM_OBSERVATIONS,
        metavar="N",
        help="Number of GPS observation points per trail",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./src/models/mlp.keras"),
        metavar="MODEL_PATH",
        help="Where to save trained model",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    pts = args.points
    mlp = model(pts)

    data_df = prepare_trail_data(str(args.input))
    y = pd.get_dummies(data_df["difficulty"], drop_first=True, dtype="int").to_numpy()

    mat = df_to_input(data_df, pts)
    print(mat.shape)
    ele_mat = mat[:, :, 2]
    print(ele_mat.shape)

    # train, test = split_dataset(ele_mat, right_size=0.2)
    # print("Train:", len(train), "| Test:", len(test))

    mlp.fit(
        mat, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1
    )
    mlp.save(args.output)

    # validate w/ known data
    # TODO hahahaha


if __name__ == "__main__":
    if K.backend() != "torch":
        raise Exception("NOT USING TORCH")

    main()
