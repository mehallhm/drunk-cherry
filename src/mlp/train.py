import os

os.environ["KERAS_BACKEND"] = "torch"

import argparse
from pathlib import Path

import pandas as pd
from keras import backend as K
from sklearn.model_selection import train_test_split

from trail_helpers import prepare_trail_data
from mlp.model import model, NUM_OBSERVATIONS
from mlp.utils import df_to_input

SEED = 42


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
        default=Path("./models/mlp.keras"),
        metavar="MODEL_PATH",
        help="Where to save trained model",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of data for the test set (default: 0.15)",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of data for the validation set (default: 0.15)",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    pts = args.points
    mlp = model(pts)

    data_df = prepare_trail_data(str(args.input))
    labels = data_df["difficulty"].values
    y = pd.get_dummies(data_df["difficulty"], drop_first=True, dtype="int").to_numpy()
    X = df_to_input(data_df, pts)

    # Split into train / val / test with stratification
    X_trainval, X_test, y_trainval, y_test, lbl_trainval, _ = train_test_split(
        X, y, labels, test_size=args.test_size, random_state=SEED, stratify=labels
    )
    val_frac = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=SEED, stratify=lbl_trainval
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    mlp.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
    )

    test_loss, test_acc = mlp.evaluate(X_test, y_test)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    mlp.save(args.output)


if __name__ == "__main__":
    if K.backend() != "torch":
        raise Exception("NOT USING TORCH")

    main()
