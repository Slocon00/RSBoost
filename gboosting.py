import numpy as np
import measures

class _GBTreeModel:
    '''A class that represents a gradient boosting tree model.'''
    def __init__(self, n_estimators: int,
                 max_depth: int,
                 lr: float,
                 alpha: float,
                 lmbda: float,
                 gamma: float,
                 measure: str):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.alpha = alpha
        self.lmbda = lmbda
        self.gamma = gamma

        if measure == 'gini':
            self.measure = measures.gini
        elif measure == 'entropy':
            self.measure = measures.entropy
        else:
            raise ValueError("Impurity measure must be either 'gini' or 'entropy'.")

        self.ensemble = []

    def fit(self, X, y):
        pass
