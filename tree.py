import numpy as np
from typing import Self

class Tree:
    ''' A class that represents a tree.'''
    def __init__(self,
                 max_depth: int = None,
                 measure: str = 'gini',
                 lmbda: float = 0.0):
        self.root = None
        self.max_depth = max_depth
        self.measure = measure
        self.lmbda = lmbda
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        '''Fit the tree to the data.'''
        #TODO implement as a depth first building algorithm. For each iter. of the loop:
        # 1. if depth > max_depth, stop and return node as leaf
        # 2. calculate the similarity score
        # 3. calculate different possible splits (implement separate methods for greedy/approximate/weighted quantile)
        # 4. for each split evaluate sim. score gain (regularized), choose the best
        # 5. IF best gain > 0, accept split, continue with left child, then right child
        # 6. IF best gain <= 0, stop and return node as leaf
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict the output of each sample of X.'''
        #TODO
        return np.zeros(X.shape[0])

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
