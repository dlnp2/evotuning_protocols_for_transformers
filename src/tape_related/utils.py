import numpy as np
from scipy.stats import spearmanr as sp_spearmanr


def spearmanr(a: np.ndarray, b: np.ndarray) -> float:
    rank_a = np.argsort(np.argsort(a)[::-1])
    rank_b = np.argsort(np.argsort(b)[::-1])
    return sp_spearmanr(rank_a, rank_b)[0]

