import numpy as np
from tree import Tree
from tqdm import tqdm # type: ignore

class XGBTreeModel:
    '''A class that represents a gradient boosting tree model 
    trained with the algorithm used by XGBoost.
    '''
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 starting_value: float = 0.5,
                 eta: float = 0.1,
                 lmbda: float = 0.0,
                 gamma: float = 0.0,
                 algorithm: str = 'exact'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.starting_value = starting_value
        self.eta = eta
        self.lmbda = lmbda
        self.gamma = gamma
        
        if algorithm == 'exact' or algorithm == 'approx':
            self.algorithm = algorithm
        else:
            raise ValueError("Splitting algorithm must be either 'exact' or 'approx'.")

        self.ensemble = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        '''Produces a fitted model.'''
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples; "
                             "X has {} and y has {} instead.".format(X.shape[0], y.shape[0]))
        
        pbar = tqdm(range(self.n_estimators), disable=not verbose)

        residuals = y - self.starting_value
        for _ in range(self.n_estimators):
            tree = Tree(max_depth=self.max_depth,
                        lmbda=self.lmbda,
                        algorithm=self.algorithm)
            tree.fit(X, residuals)

            # TODO tree pruning using gamma?
            residuals = self.eta * tree.predict(X)
            self.ensemble.append(tree)

            pbar.update(1)
        pbar.close()

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predicts the output of each sample of X.'''
        pred = np.zeros(X.shape[0])
        for tree in self.ensemble:
            pred += tree.predict(X)
        return pred