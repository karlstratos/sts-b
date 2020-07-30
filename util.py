import numpy as np


def cos_numpy(u, v):
    return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
