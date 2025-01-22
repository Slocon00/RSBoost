import numpy as np
from tree import Tree
from tqdm import tqdm # type: ignore

class _XGBTreeModel:
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
                 loss: str = 'mse',
                 algorithm: str = 'exact'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.starting_value = starting_value
        self.eta = eta
        self.lmbda = lmbda
        self.gamma = gamma
        
        if loss == 'mse' or loss == 'logistic':
            self.loss = loss
        else:
            raise ValueError("Loss function must be either 'mse' or 'logistic'.")

        if algorithm == 'exact' or algorithm == 'approx':
            self.algorithm = algorithm
        else:
            raise ValueError("Splitting algorithm must be either 'exact' or 'approx'.")

        self.ensemble = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        '''Produce a fitted model.'''
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples; "
                             "X has {} and y has {} instead.".format(X.shape[0], y.shape[0]))
        
        pbar = tqdm(range(self.n_estimators), disable=not verbose)

        pred = np.array([self.starting_value] * X.shape[0])
        for _ in range(self.n_estimators):
            tree = Tree(max_depth=self.max_depth,
                        lmbda=self.lmbda,
                        gamma=self.gamma,
                        loss=self.loss,
                        algorithm=self.algorithm)
            tree.fit(X, (y - pred), pred)

            if self.loss == 'logistic':
                # convert pred to log odds, add to prediction,
                # then convert back to probability
                pred = np.log(pred/(1-pred))
                pred += self.eta * tree.predict(X)
                pred = 1/(1 + np.exp(-pred))
            else:
                pred += self.eta * tree.predict(X)
            
            self.ensemble.append(tree)
            pbar.update(1)
        pbar.close()

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Predict the output of each sample of X.'''
        # Output is starting value + sum of predictions of each tree
        pred = np.array([self.starting_value] * X.shape[0])
        for tree in self.ensemble:
            if self.loss == 'logistic':
                # convert pred to log odds, add to prediction,
                # then convert back to probability
                pred = np.log(pred/(1-pred))
                pred += self.eta * tree.predict(X)
                pred = 1/(1 + np.exp(-pred))
            else:
                pred += self.eta * tree.predict(X)
        return pred


class XGBTreeClassifier(_XGBTreeModel):
    '''A class that represents a gradient boosting tree classifier 
    trained with the algorithm used by XGBoost.
    '''
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 starting_value: float = 0.5,
                 eta: float = 0.1,
                 lmbda: float = 0.0,
                 gamma: float = 0.0,
                 loss: str = 'logistic',
                 algorithm: str = 'exact'):
        super().__init__(n_estimators,
                         max_depth,
                         starting_value,
                         eta,
                         lmbda,
                         gamma,
                         loss,
                         algorithm)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        '''Produce a fitted model.'''
        super().fit(X, y, verbose)

    def predict(self, X):
        '''Predict class labels for each sample in X.'''
        pred = super().predict(X)
        pred = np.round(pred) + 0  # the + 0 "fixes" negative 0s
        return pred


class XGBTreeRegressor(_XGBTreeModel):
    '''A class that represents a gradient boosting tree regressor 
    trained with the algorithm used by XGBoost.
    '''
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 starting_value: float = 0.5,
                 eta: float = 0.1,
                 lmbda: float = 0.0,
                 gamma: float = 0.0,
                 loss: str = 'mse',
                 algorithm: str = 'exact'):
        super().__init__(n_estimators,
                         max_depth,
                         starting_value,
                         eta,
                         lmbda,
                         gamma,
                         loss,
                         algorithm)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
        '''Produce a fitted model.'''
        super().fit(X, y, verbose)

    def predict(self, X):
        '''Predict regression value for each sample in X.'''
        return super().predict(X)