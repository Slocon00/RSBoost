import numpy as np
from typing import Self

class Tree:
    ''' A class that represents a tree.'''
    def __init__(self, max_depth: int = None):
        self.root = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, lmbda: float = 0.0, alpha: float = 0.0):
        '''Fit the tree to the data.'''
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict the output of each sample of X.'''
        pass

class Node:
    '''A class that represents a node of a Tree.'''
    def __init__(self, feature: int, threshold: float | str):
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
    
    def add_left(self, node: Self):
        self.left = node
    
    def add_right(self, node: Self):
        self.right = node
