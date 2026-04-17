import numpy as np
import pandas as pd


def downsample_to_length(matrix: np.ndarray, pts: int) -> np.ndarray:
    """
    Downsample or zero-pad a (N, 3) array to exactly `pts` rows

    :param matrix: matrix of trail obs to downsample
    :param pts: number of input points
    :return: matrix of observations (n, pts, 3)
    """
    n = matrix.shape[0]
    if n >= pts:
        indices = np.linspace(0, n - 1, pts, dtype=int)
        return matrix[indices]
    padded = np.zeros((pts, matrix.shape[1]))
    padded[:n] = matrix
    return padded


def df_to_input(df: pd.DataFrame, pts: int) -> np.ndarray:
    """
    Convert trail DataFrame to a (num_trails, pts, 3) array of [lon, lat, ele]

    :param df: trail DataFrame
    :param pts: number of input points
    :return: matrix of observations (n, pts, 3)
    """
    trails = []
    for _, row in df.iterrows():
        trail = np.column_stack([row["longitude"], row["latitude"], row["elevation"]])
        trails.append(downsample_to_length(trail, pts))
    return np.stack(trails)
