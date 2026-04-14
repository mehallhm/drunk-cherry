"""
Helpers to load trail data
"""

import pandas as pd

KNOWN_DIFFICULTIES = [
    "Easy",
    "Easy/Intermediate",
    "Intermediate",
    "Intermediate/Difficult",
    "Difficult",
    "Very Difficult",
]

DIFFICULTY_MAP = {
    "Easy": "Easy",
    "Easy/Intermediate": "Intermediate",
    "Intermediate": "Intermediate",
    "Intermediate/Difficult": "Difficult",
    "Difficult": "Difficult",
    "Very Difficult": "Very Difficult",
}


def load_trail_data(path: str) -> pd.DataFrame:
    """
    Load trail CSV (one row per GPS point, grouped by trail_id) and
    return a DataFrame with one row per trail, where latitude, longitude,
    and elevation are aggregated into lists.

    :return: DataFrame with one row per trail
    """
    df = pd.read_csv(path, encoding="utf-8", compression="zip")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    trail_attrs = [
        "difficulty",
        "rating",
        "length",
        "elevation_gain",
        "elevation_loss",
        "average_grade",
        "max_grade",
    ]

    grouped = (
        df.groupby("trail_id", sort=False)
        .agg(
            **{col: (col, "first") for col in trail_attrs},
            latitude=("latitude", list),
            longitude=("longitude", list),
            elevation=("elevation", list),
        )
        .reset_index()
    )

    return grouped


def clean_trails(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    # remove nans from difficulty column
    df.dropna(subset=["difficulty"])
    return df


def filter_known_difficulties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose difficulty is in KNOWN_DIFFICULTIES

    :param df: trails dataframe
    :return: dataframe with only "known" difficulties
    """
    return df[df["difficulty"].isin(KNOWN_DIFFICULTIES)].reset_index(drop=True)


def consolidate_difficulties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map 6 difficulty classes down to 4

    :param df: trails dataframe
    :return: df with configured difficulty classes
    """
    df = df.copy()
    df["difficulty"] = df["difficulty"].map(DIFFICULTY_MAP)
    return df


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsample each difficulty class to the size of the smallest class

    :param df: trails dataframe
    :return: df with balanced difficulty class sizes
    """
    min_count = df["difficulty"].value_counts().min()
    balanced = pd.concat(
        group.sample(n=min_count, random_state=42)
        for _, group in df.groupby("difficulty")
    )
    return balanced.reset_index(drop=True)


def prepare_trail_data(path: str) -> pd.DataFrame:
    """
    Full pipeline: load, filter, consolidate, and balance trail data

    :param path: path to trail data
    :return: fully pre-processed trail data
    """
    df = load_trail_data(path)
    df = clean_trails(df)
    df = filter_known_difficulties(df)
    df = consolidate_difficulties(df)
    df = balance_classes(df)
    return df
