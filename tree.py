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
                 algorithm: str = 'exact',
                 epsilon: float = 0.1):
        self.root = None
        self.max_depth = max_depth
        self.lmbda = lmbda
        self.gamma = gamma
        self.algorithm = algorithm
        self.epsilon = epsilon
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            gradients: np.ndarray,
            hessians: np.ndarray):
        '''Fit the tree to the data.'''
        self.root = self._build_tree(X, y, gradients, hessians, 0)

    def _build_tree(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    gradients: np.ndarray,
                    hessians: np.ndarray,
                    depth: int):
        '''Build the tree recursively.'''
        if depth == self.max_depth:
            pred = np.sum(gradients)/(np.sum(hessians) + self.lmbda)
            return _Node(None, pred)
        
        split = self._find_best_split(X, y, gradients, hessians)
        if split is None:
            # No split improves the gain
            pred = np.sum(gradients)/(np.sum(hessians) + self.lmbda)
            return _Node(None, pred)

        node = _Node(*split)
        left_child = self._build_tree(X[X[:, node.feature] <= node.threshold],
                                      y[X[:, node.feature] <= node.threshold],
                                      gradients[X[:, node.feature] <= node.threshold],
                                      hessians[X[:, node.feature] <= node.threshold],
                                      depth + 1)
        right_child = self._build_tree(X[X[:, node.feature] > node.threshold],
                                       y[X[:, node.feature] > node.threshold],
                                       gradients[X[:, node.feature] > node.threshold],
                                       hessians[X[:, node.feature] > node.threshold],
                                       depth + 1)
        node.add_left(left_child)
        node.add_right(right_child)

        return node

    def _find_best_split(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         gradients: np.ndarray,
                         hessians: np.ndarray,
                         algorithm: str = 'exact') -> tuple[int, float] | None:
        '''Return the best split for the data, using the specified algorithm.'''
        if algorithm == 'exact': return self._split_exact(X, y, gradients, hessians)
        if algorithm == 'approx': return self._split_approx(X, y, gradients, hessians)

    def _split_exact(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     gradients: np.ndarray,
                     hessians: np.ndarray) -> tuple[int, float] | None:
        '''Find the best split that produces a positive similarity score gain
        using the exact algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        best_split = None
        parent_sim = self._similarity_score(y, gradients, hessians)

        for i in range(X.shape[1]):
            feat_vals = np.unique(X[:,i])

            # splits are midpoints between feat values
            splits = (feat_vals[:-1] + feat_vals[1:]) / 2
            for split in splits:
                left_y = y[X[:,i] <= split]
                right_y = y[X[:,i] > split]

                left_g = gradients[X[:,i] <= split]
                right_g = gradients[X[:,i] > split]
                left_h = hessians[X[:,i] <= split]
                right_h = hessians[X[:,i] > split]

                # if either leaf becomes empty skip
                if len(left_y) < 1 or len(right_y) < 1:
                    continue

                left_sim = self._similarity_score(left_y, left_g, left_h)
                right_sim = self._similarity_score(right_y, right_g, right_h)

                if (left_sim + right_sim - parent_sim) - self.gamma > 0.0:
                    # gain is positive, better than last best
                    best_split = (i, split)
                    parent_sim = (left_sim + right_sim) - self.gamma

        return best_split

    def _split_approx(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      gradients: np.ndarray,
                      hessians: np.ndarray) -> tuple[int, float] | None:
        '''Find the best split that produces a positive similarity score gain
        using the approximate algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        # S = np.unique(X)
        # summary = (S,
        #           np.array([self._r_minus(X, hessians, x) for x in S]),
        #           np.array([self._r_plus(X, hessians, x) for x in S]),
        #           np.array([self._omega(X, hessians, x) for x in S]))
        best_split = None
        parent_sim = self._similarity_score(y, gradients, hessians)

        for i in range(X.shape[1]):
            # find splits S = {s_1, s_2, ... , s_n} s.t.
            # |rank(s_i) - rank(s_i+1)| < epsilon
            splits = np.quantile(X[:,i],
                                 np.linspace(0, 1, 1/self.epsilon),
                                 weights=hessians)
            
            for split in splits:
                left_y = y[X[:,i] <= split]
                right_y = y[X[:,i] > split]

                left_g = gradients[X[:,i] <= split]
                right_g = gradients[X[:,i] > split]
                left_h = hessians[X[:,i] <= split]
                right_h = hessians[X[:,i] > split]

                # if either leaf becomes empty skip
                if len(left_y) < 1 or len(right_y) < 1:
                    continue

                left_sim = self._similarity_score(left_y, left_g, left_h)
                right_sim = self._similarity_score(right_y, right_g, right_h)

                if (left_sim + right_sim - parent_sim) - self.gamma > 0.0:
                    # gain is positive, better than last best
                    best_split = (i, split)
                    parent_sim = (left_sim + right_sim) - self.gamma

        return best_split

    def _r_minus(self, X: np.ndarray, h: np.ndarray, x: np.ndarray) -> float:
        '''Return the rank of x, calculated as the sum of weights assigned to
        vectors in X that are strictly smaller than x.
        '''
        return np.sum(h[X < x]) if np.any(X < x) else 0

    def _r_plus(self, X: np.ndarray, h: np.ndarray, x: np.ndarray) -> float:
        '''Return the rank of x, calculated as the sum of weights assigned to
        vectors in X that are smaller than or equal to x.
        '''
        return np.sum(h[X <= x]) if np.any(X <= x) else 0
    
    def _omega(self, X: np.ndarray, h: np.ndarray, x: np.ndarray) -> float:
        '''Return the weight of x, calculated as the sum of weights assigned to
        vectors in X that are equal to x.
        '''
        return np.sum(h[X == x]) if np.any(X == x) else 0

    def _merge_summaries(self, left: tuple, right: tuple) -> tuple:
        '''Merge the two summaries in a single one.'''
        S = np.unique(np.concatenate((left[0], right[0])))
        r_minus = None
        r_plus = None
        omega = None
        
        return (S, r_minus, r_plus, omega)

    def _prune_summary(self, summary: tuple) -> tuple:
        '''Prune the summary, reducing its size.'''
        # TODO
        pass

    def _similarity_score(self, y: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> float:
        '''Return the similarity score of the data.'''
        return np.sum(gradients)**2 / (np.sum(hessians) + self.lmbda)

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


