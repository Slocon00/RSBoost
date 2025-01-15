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
        self._build_tree(X, y, 0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        '''Build the tree recursively.'''
        #TODO implement as a depth first building algorithm. For each iter:
        # 1. if depth > max_depth, stop and return node as leaf
        # 2. calculate the similarity score
        # 3. calculate different possible splits (implement separate methods for greedy/approximate)
        # 4. for each split evaluate sim. score gain (regularized), choose the best
        # 5. IF best gain > 0, accept split, continue with left child, then right child
        # 6. IF best gain <= 0, stop and return node as leaf
        pass

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, algorithm: str = 'exact'):
        if algorithm == 'exact': return self._split_exact(X, y)
        if algorithm == 'approx': return self._split_approx(X, y)

    def _split_exact(self, X: np.ndarray, y: np.ndarray):
        pass

    def _split_approx(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict the output of each sample of X.'''
        pred = np.ndarray(X.shape[0])
        
        for i in range(X.shape[0]):
            curr = self.root
            while True:
                if curr.left is None and curr.right is None:
                    pred[i] = curr.threshold
                    break
                print(X[i][curr.feature])
                if (X[i][curr.feature] <= curr.threshold):
                    curr = curr.left
                else:
                    curr = curr.right

        return pred
    

class _Node:
    '''A class that represents a node of a Tree.

    A split Node has both a left and a right child, as well as a feature index
    and a threshold, such that samples with feature value <= threshold are
    assigned to the left child, and samples with feature value > threshold are
    assigned to the right child.

    Leaves are represented by Nodes with both left and right set to None,
    as well as feature, and where the threshold is set to the predicted value
    for that leaf.
    '''
    def __init__(self, feature: int, threshold: float | str):
        self.feature = feature
        self.threshold = threshold
        self.left = None
        self.right = None
    
    def add_left(self, node: Self):
        self.left = node
    
    def add_right(self, node: Self):
        self.right = node
