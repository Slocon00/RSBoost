import numpy as np
from tree import Tree
import measures
from tqdm import tqdm # type: ignore

class XGBTreeModel:
    '''A class that represents a gradient boosting tree model 
    trained with the algorithm used by XGBoost.
    '''
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 starting_value: float = 0.5,
                 lr: float = 0.1,
                 lmbda: float = 0.0,
                 gamma: float = 0.0,
                 measure: str = 'gini'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.starting_value = starting_value
        self.lr = lr
        self.lmbda = lmbda
        self.gamma = gamma

        if measure == 'gini':
            self.measure = measures.gini
        elif measure == 'entropy':
            self.measure = measures.entropy
        else:
            raise ValueError("Impurity measure must be either 'gini' or 'entropy'.")

        self.ensemble = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        '''Produces a fitted model.'''
        pbar = tqdm(range(self.n_estimators), disable=not verbose)

        residuals = y - self.starting_value
        for _ in range(self.n_estimators):
            tree = Tree(max_depth=self.max_depth,
                        measure=self.measure,
                        lmbda=self.lmbda)
            tree.fit(X, residuals)

            # TODO tree pruning using gamma?
            residuals = tree.predict(X)
            self.ensemble.append(tree)

            pbar.update(1)
        pbar.close()

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predicts the output of each sample of X.'''
        pred = np.zeros(X.shape[0])
        for tree in self.ensemble:
            pred += tree.predict(X)
        return pred