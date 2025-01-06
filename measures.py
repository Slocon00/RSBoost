import numpy as np

def gini(y: np.ndarray) -> float:
    return 1 - np.sum((np.bincount(y)/len(y))**2)

def entropy(y: np.ndarray) -> float:
    p = np.bincount(y)/len(y)
    return - np.sum(p * np.log2(p))
