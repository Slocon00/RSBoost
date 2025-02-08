import numpy as np
from typing import Self, Optional
from dataclasses import dataclass


@dataclass
class _ThresholdT:
    '''
    Represents a threshold for an internal node.

    Attributes:
        feature (int): represents the feature for which to compare.
        value (float): represents the value itself for which to compare..
    '''
    feature: int
    value: float

@dataclass
class _RSNode:
    '''
    An abstract class that represents a node of a Gradient Boost Tree.
    A node can be an internal one or a leaf. Since different type of nodes
    contains different paramenters, they are implemented as different subclasses
    of this class.
    '''
    pass

@dataclass
class _ChildrenT:
    '''
    Children are represented as optional int values containing 
    the index of the associated child in the array of nodes
    representing the tree.
    '''
    left: Optional[int]
    right: Optional[int]

    @classmethod
    def __init__(self):
        '''
        Creates an empty pair
        '''
        self.left = None
        self.right = None

@dataclass
class _Leaf(_RSNode):
    '''
    Represents a leaf of a Gradient Boost Tree.
    
    Attributes:
        prediction(float): TODO mettici qui qualcosa
    '''
    prediction: float

    @classmethod
    def is_leaf(self):
        True

@dataclass
class _Internal(_RSNode):
    '''
    Represents a internal node of a Gradient Boost Tree.
    
    Attributes:
        children: a ChildrenT value containing the indices of the node's children.
        threshold: a ThresholdT value describing how we spit in the node.
    '''
    children : _ChildrenT
    threshold: _ThresholdT

    @classmethod
    def is_leaf(self):
        False

@dataclass(frozen=True) # frozen=True is here to make it immutable
class _AlgorithmT:
    """Represent the the type of split finding algorithm employed by the RSTree."""
    
    @classmethod
    def from_str(cls, name: str, epsilon: Optional[float] = None) -> "_AlgorithmT":
        name = name.lower().strip()
        if name == "exact":
            return _Exact()
        elif name == "approximate":
            if epsilon is None:
                raise ValueError("An 'epsilon' value is required for the approximate algorithm.")
            return _Approximate(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown algorithm type: {name}")

@dataclass(frozen=True) # frozen=True is here to make it immutable
class _Exact(_AlgorithmT):
    """
    Exact algorithm that enumerates over all possible splits
    on all the features.
    """

@dataclass(frozen=True) # frozen=True is here to make it immutable
class _Approximate(_AlgorithmT):
    """
    Approximate algorithm that first proposes candidate splits
    based on percentiles of the feature distribution.
    
    Attributes:
        epsilon (float): A hyperparameter such that the number of percentiles is 1/epsilon.
    """
    epsilon: float

@dataclass
class _Statistics:
    '''
    Contains all the data necessary for constructing a RSTree:
    
    Attributes:
        inputs: the values of the training instances. It is containes as a 2D matrix.
        targets: the targets of the training instances.
        gradients: TODO description here
        hessians: TODO description here
    '''
    inputs: np.ndarray
    targets: np.ndarray
    gradients: np.ndarray
    hessians: np.ndarray

class RSTree:
    '''
    A simple re-implementation of a XGBoost Tree.
    The tree is a Binary Tree.
    
    Attributes:
        nodes: the vector of nodes representing the tree's structure.
        max_depth : a limit indicating the max_depth the tree can reach during construction.
        lmbda: a constant hyperparameter TODO Add better description.
        gamma: a constant hyperparameter TODO Add better description.
        algorithm: indicates the type of split algorithm used for constructing the tree.
    '''
    def __init__(self,
                 max_depth: Optional[int] = None,
                 lmbda: float = 0.0,
                 gamma: float = 0.0,
                 algorithm: str = 'exact',
                 epsilon: float = 0.0):
        self.nodes: list[_RSNode] = []
        self.max_depth = max_depth
        self.lmbda = lmbda
        self.gamma = gamma
        self.algorithm = _AlgorithmT.from_str(algorithm, epsilon)

    def fit(self,
                    inputs: np.ndarray,
                    targets: np.ndarray,
                    gradients: np.ndarray,
                    hessians: np.ndarray,):
        '''
        Constructs recursively the Boost Tree.

        Parameters:
            inputs: the matrix containing the input values of the instances
            targets: the matrix containing the target values of the instances
            gradients: the matrix contaning the gradients (prima derivata)
            hessians: the matrix containing the second derivative (TODO VEDI COME SI DICE LA SECONDA DERIVATA)
        '''
        # Compact the given numpy arrays into a _Statistics struct
        data = _Statistics(inputs, targets, gradients, hessians)
        self._build_tree(data, 0)

    def _build_tree(self,
                    data: _Statistics,
                    depth: int):
        '''
        Constructs recursively the Boost Tree.

        Parameters:
            depth: the depth of the current node being constructed. If it is over the max_depth, it will construct a leaf instead and terminate the visit.
            data: the set of datas used to construct the node.

        Returns:
            the index of the node in the array
        '''
        # Constructing an internal node
        if depth != self.max_depth:
            
            # compute the best split
            split = None
            match self.algorithm:
                case _Exact():
                    split = self._split_exact(data)
                case _Approximate(epsilon = epsilon):
                    split = self._split_approx(data, epsilon)
                case _:
                    raise ValueError("Unknown Split Algorithm variant")
            
            # if the split is None then construct a Leaf,
            # otherwise continue
            if split is not None:
                node = _Internal(_ChildrenT(), split)
                self.nodes.append(node)
                node_idx = len(self.nodes) - 1
                # Computing children
                left_child_idx = self._build_tree(
                    _Statistics(data.inputs[data.inputs[:, node.threshold.feature] <= node.threshold.value],
                                data.targets[data.inputs[:, node.threshold.feature] <= node.threshold.value],
                                data.gradients[data.inputs[:, node.threshold.feature] <= node.threshold.value],
                                data.hessians[data.inputs[:, node.threshold.feature] <= node.threshold.value]
                                ),
                    depth+1)
                
                right_child_idx = self._build_tree(
                    _Statistics(data.inputs[data.inputs[:, node.threshold.feature] > node.threshold.value],
                                data.targets[data.inputs[:, node.threshold.feature] > node.threshold.value],
                                data.gradients[data.inputs[:, node.threshold.feature] > node.threshold.value],
                                data.hessians[data.inputs[:, node.threshold.feature] > node.threshold.value]
                                ),
                    depth+1)
                
                # Add children
                self.nodes[node_idx].children.left = left_child_idx
                self.nodes[node_idx].children.right = right_child_idx

                return node_idx
        ## Construct a leaf
        pred = np.sum(data.gradients)/(np.sum(data.hessians) + self.lmbda)
        self.nodes.append(_Leaf(prediction=pred))
        node_idx = len(self.nodes) - 1
        return node_idx
    
    def _split_exact(self,
                     data: _Statistics,
                     ) -> Optional[_ThresholdT]:
        '''
        Find the best split that produces a positive similarity score gain
        using the exact algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        best_split = None
        parent_sim = self._similarity_score(data.gradients, data.hessians)

        for i in range(data.inputs.shape[1]):
            feat_vals = np.unique(data.inputs[:, i])

            # splits are midpoints between feat values
            splits = (feat_vals[:-1] + feat_vals[1:]) / 2
            for split in splits:
                # TODO See if you can optimize the computation of left_something and right_something doing them together
                left_target = data.targets[data.inputs[:,i] <= split]
                right_target = data.targets[data.inputs[:,i] > split]

                left_gradients = data.gradients[data.inputs[:,i] <= split]
                right_gradients = data.gradients[data.inputs[:,i] > split]
                
                left_hessians = data.hessians[data.inputs[:,i] <= split]
                right_hessians = data.hessians[data.inputs[:,i] > split]

                # if either leaf becomes empty skip
                if len(left_target) < 1 or len(right_target) < 1:
                    continue

                left_sim = self._similarity_score(left_gradients, left_hessians)
                right_sim = self._similarity_score(right_gradients, right_hessians)

                if (left_sim + right_sim - parent_sim) - self.gamma > 0.0:
                    # gain is positive, better than last best
                    best_split = _ThresholdT(i, split)
                    parent_sim = (left_sim + right_sim) - self.gamma
        
        return best_split
    
    def _split_approx (self,
                       data: _Statistics,
                       epsilon: float) -> Optional[_ThresholdT]:
        
        '''
        Find the best split that produces a positive similarity score gain
        using the approximate algorithm, returning a tuple of feature index and
        threshold. If no such split exists, return None.
        '''
        best_split = None
        parent_sim = self._similarity_score(data.gradients, data.hessians)

        for i in range(data.inputs.shape[1]):
            feat_vals = np.unique(data.inputs[:, i])

            S = (feat_vals,
                 np.array([self._r_minus(data.inputs, data.hessians, x) for x in feat_vals]), #TODO here i substituted x in S with x in feat_vals, check if it is correct
                 np.array([self._r_plus(data.inputs, data.hessians, x) for x in feat_vals]), #TODO here i substituted x in S with x in feat_vals, check if it is correct
                 np.array([self._omega(data.inputs, data.hessians, x) for x in feat_vals]) #TODO here i substituted x in S with x in feat_vals, check if it is correct
                 ) 

            # epsilon is the "approximation factor", s.t. the difference in rank
            # between two splits is less than epsilon
            start, end = np.min(S[3]), np.max(S[3])
            splits = []
            for d in np.linspace(start, end, int((end - start)/epsilon)):
                splits.append(self._query(S, d))  

            for split in splits:
                # TODO See if you can optimize the computation of left_something and right_something doing them together
                left_target = data.targets[data.inputs[:,i] <= split]
                right_target = data.targets[data.inputs[:,i] > split]

                left_gradients = data.gradients[data.inputs[:,i] <= split]
                right_gradients = data.gradients[data.inputs[:,i] > split]
                
                left_hessians = data.hessians[data.inputs[:,i] <= split]
                right_hessians = data.hessians[data.inputs[:,i] > split]

                # if either leaf becomes empty skip
                if len(left_target) < 1 or len(right_target) < 1 :
                    continue

                left_sim = self._similarity_score(left_gradients, left_hessians)
                right_sim = self._similarity_score(right_gradients, right_hessians)

                if (left_sim + right_sim - parent_sim) - self.gamma > 0.0:
                    # gain is positive, better than last best
                    best_split = _ThresholdT(i, split)
                    parent_sim = (left_sim + right_sim) - self.gamma
        
        return best_split
        
    
    def _r_minus(self, X: np.ndarray, h: np.ndarray, x: np.ndarray) -> float:
        '''
        Return the rank of x, calculated as the sum of weights assigned to
        vectors in X that are strictly smaller than x.
        '''
        return np.sum(h[X < x]) if np.any(X < x) else 0

    def _r_plus(self, X: np.ndarray, h: np.ndarray, x: np.ndarray) -> float:
        '''
        Return the rank of x, calculated as the sum of weights assigned to
        vectors in X that are smaller than or equal to x.
        '''
        return np.sum(h[X <= x]) if np.any(X <= x) else 0
    
    def _omega(self, X: np.ndarray, h: np.ndarray, x: np.ndarray) -> float:
        '''
        Return the weight of x, calculated as the sum of weights assigned to
        vectors in X that are equal to x.
        '''
        return np.sum(h[X == x]) if np.any(X == x) else 0
    
    def _query(self, S: tuple, d: float):
        '''
        Given the summary S and the value d, find the record x in S that is
        the d*100th percentile of the data.
        '''
        if d < 0 or d > 1:
            raise ValueError("d must be between 0. and 1.")

        values, r_minus, r_plus, omega = S
        if d < (r_minus[0] + r_plus[0]) / 2: return values[0]
        if d >= (r_minus[-1] + r_plus[-1]) / 2: return values[-1]

        for i in range(len(values)-1):
            if (r_minus[i] + r_plus[i])/2 <= d < (r_minus[i+1] + r_plus[i+1])/2:
                if 2*d < r_minus[i] + omega[i] + r_plus[i+1] - omega[i+1]:
                    return values[i]
                else:
                    return values[i+1]
                

    def _similarity_score(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        '''Return the similarity score of the data.'''
        return np.sum(gradients)**2 / (np.sum(hessians) + self.lmbda)
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        '''Predict the output of each sample of input.'''        
        pred = np.zeros(input.shape[0])
        
        for i in range(input.shape[0]):
            curr_idx = 0
            while True:
                match self.nodes[curr_idx]:
                    case _Internal(children, threshold):
                        if (input[i][threshold.feature] <= threshold.value):
                            curr_idx = children.left
                        else:
                            curr_idx = children.right
                    case _Leaf(prediction):
                        pred[i] = prediction
                        break

        return pred





                
        





    