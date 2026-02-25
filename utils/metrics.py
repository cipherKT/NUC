import numpy as np


def spatial_mad(frame):
    mean_val = np.mean(frame)
    return np.mean(np.abs(frame - mean_val))


def col_mad(frame):
    col_mean = np.mean(frame, axis=0)
    return np.mean(np.abs(frame - col_mean))


def row_mad(frame):
    row_mean = np.mean(frame, axis=1, keepdims=True)
    return np.mean(np.abs(frame - row_mean))
