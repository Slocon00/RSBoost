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
    def __init__(self, feature: int | None, threshold: float | str):
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
                 gamma: float = 0.0,
                 loss: str = 'mse',
                 algorithm: str = 'exact'):
        self.root = None
        self.max_depth = max_depth
        self.lmbda = lmbda
        self.gamma = gamma
        self.loss = loss
        self.algorithm = algorithm
    
    def fit(self, X: np.ndarray, y: np.ndarray, probs: np.ndarray):
        '''Fit the tree to the data.'''
        self.root = self._build_tree(X, y, probs, 0)

    def _build_tree(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    probs: np.ndarray,
                    depth: int):
        '''Build the tree recursively.'''

        if depth == self.max_depth:
            if self.loss == 'mse':
                value = np.sum(y)/(len(y)+self.lmbda)
            elif self.loss == 'logistic':
                value = np.sum(y)/(np.sum(probs*(1-probs)) + self.lmbda)
            return _Node(None, value)
        
        split = self._find_best_split(X, y, probs)
        if split is None:
            # No split improves the gain
            if self.loss == 'mse':
                value = np.sum(y)/(len(y)+self.lmbda)
            elif self.loss == 'logistic':
                value = np.sum(y)/(np.sum(probs*(1-probs)) + self.lmbda)
            return _Node(None, value)

        node = _Node(*split)
        left_child = self._build_tree(X[X[:, node.feature] <= node.threshold],
                                      y[X[:, node.feature] <= node.threshold],
                                      probs[X[:, node.feature] <= node.threshold],
                                      depth + 1)
        right_child = self._build_tree(X[X[:, node.feature] > node.threshold],
                                       y[X[:, node.feature] > node.threshold],
                                       probs[X[:, node.feature] > node.threshold],
                                       depth + 1)
        node.add_left(left_child)
        node.add_right(right_child)

        return node

    def _find_best_split(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         probs: np.ndarray,
                         algorithm: str = 'exact') -> tuple[int, float] | None:
        '''Return the best split for the data, using the specified algorithm.'''
        if algorithm == 'exact': return self._split_exact(X, y, probs)
        if algorithm == 'approx': return self._split_approx(X, y, probs)

    def _split_exact(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     probs: np.ndarray
                     ) -> tuple[int, float] | None:
        '''Find the best split that produces a positive similarity score gain
        using the exact algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        best_split = None
        parent_sim = self._similarity_score(y, probs)

        for i in range(X.shape[1]):
            # TODO categorical values?
            feat_vals = np.unique(X[:,i])

            # splits are midpoints between feat values
            splits = (feat_vals[:-1] + feat_vals[1:]) / 2
            for split in splits:
                left_y = y[X[:,i] <= split]
                right_y = y[X[:,i] > split]
                left_pred = probs[X[:,i] <= split]
                right_pred = probs[X[:,i] > split]

                # if either leaf becomes empty skip
                if len(left_y) < 1 or len(right_y) < 1:
                    continue

                left_sim = self._similarity_score(left_y, left_pred)
                right_sim = self._similarity_score(right_y, right_pred)

                if (left_sim + right_sim - parent_sim) - self.gamma > 0.0:
                    # gain is positive, better than last best
                    best_split = (i, split)
                    parent_sim = (left_sim + right_sim) - self.gamma

        return best_split

    def _split_approx(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      probs: np.ndarray
                      ) -> tuple[int, float] | None:
        '''Find the best split that produces a positive similarity score gain
        using the approximate algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        # TODO
        return None

    def _similarity_score(self, y: np.ndarray, probs: np.ndarray) -> float:
        '''Return the similarity score of the data.'''
        if self.loss == 'mse':
            # gi is y, hi is 1
            return np.sum(y)**2 / (len(y) + self.lmbda)
        if self.loss == 'logistic':
            return np.sum(y)**2 / (np.sum(probs*(1-probs)) + self.lmbda)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict the output of each sample of X.'''        
        pred = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            curr = self.root
            while True:
                if curr.feature is None:
                    pred[i] = curr.threshold
                    break

                if (X[i][curr.feature] <= curr.threshold):
                    curr = curr.left
                else:
                    curr = curr.right

        return pred


