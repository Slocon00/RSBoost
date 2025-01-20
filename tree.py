import numpy as np
from typing import Self


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


class Tree:
    ''' A class that represents a tree.'''
    def __init__(self,
                 max_depth: int = None,
                 lmbda: float = 0.0,
                 algorithm: str = 'exact'):
        self.root = None
        self.max_depth = max_depth
        self.lmbda = lmbda
        self.algorithm = algorithm
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        '''Fit the tree to the data.'''
        self.root = self._build_tree(X, y, 0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        '''Build the tree recursively.'''

        if depth == self.max_depth:
            return _Node(None, np.mean(y))
        
        split = self._find_best_split(X, y)
        if split is None:
            # No split improves the gain
            return _Node(None, np.mean(y))

        node = _Node(*split)
        left_child = self._build_tree(X[X[:, node.feature] <= node.threshold],
                                      y[X[:, node.feature] <= node.threshold],
                                      depth + 1)
        right_child = self._build_tree(X[X[:, node.feature] > node.threshold],
                                       y[X[:, node.feature] > node.threshold],
                                       depth + 1)
        node.add_left(left_child)
        node.add_right(right_child)

        return node

    def _find_best_split(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         algorithm: str = 'exact') -> tuple[int, float] | None:
        '''Return the best split for the data, using the specified algorithm.'''
        # 3. calculate different possible splits (implement separate methods for greedy/approximate)
        # 4. for each split evaluate sim. score gain (regularized), choose the best
        if algorithm == 'exact': return self._split_exact(X, y)
        if algorithm == 'approx': return self._split_approx(X, y)

    def _split_exact(self,
                     X: np.ndarray,
                     y: np.ndarray
                     ) -> tuple[int, float] | None:
        '''Find the best split that produces a positive similarity score gain
        using the exact algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        best_split = None
        parent_sim = self.similarity_score(y)

        for i in range(X.shape[1]):
            # TODO categorical values?
            feat_vals = np.unique(X[:,i])

            # splits are midpoints between feat values
            splits = (feat_vals[:-1] + feat_vals[1:]) / 2
            for split in splits:
                left = y[X[:,i] <= split]
                right = y[X[:,i] > split]

                left_sim = self.similarity_score(left)
                right_sim = self.similarity_score(right)

                if (left_sim + right_sim - parent_sim) > 0:
                    # gain is positive
                    best_split = (i, split)
                    parent_sim = left_sim + right_sim
            
        return best_split

    def _split_approx(self,
                      X: np.ndarray,
                      y: np.ndarray
                      ) -> tuple[int, float] | None:
        '''Find the best split that produces a positive similarity score gain
        using the approximate algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        # TODO
        return None

    def similarity_score(self, y: np.ndarray) -> float:
        '''Return the similarity score of the data.'''
        return np.sum(y)**2 / (len(y) + self.lmbda)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict the output of each sample of X.'''
        pred = np.ndarray(X.shape[0])
        
        for i in range(X.shape[0]):
            curr = self.root
            while True:
                if curr.left is None and curr.right is None:
                    pred[i] = curr.threshold
                    break

                if (X[i][curr.feature] <= curr.threshold):
                    curr = curr.left
                else:
                    curr = curr.right

        return pred
    


