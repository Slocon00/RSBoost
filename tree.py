import numpy as np
from typing import Optional
from dataclasses import dataclass
from collections import deque



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

@dataclass
class _ChildrenT:
    '''
    Children are represented as optional int values containing 
    the index of the associated child in the array of nodes
    representing the tree.
    '''
    left: Optional[int]
    right: Optional[int]

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

@dataclass
class _Internal(_RSNode):
    '''
    Represents a internal node of a Gradient Boost Tree.
    
    Attributes:
        children: a ChildrenT value containing the indices of the node's children.
        threshold: a ThresholdT value describing how we spit in the node.
    '''
    threshold: _ThresholdT

class _Dummy(_RSNode):
    '''
    Represents a Dummy node. This node has no semantic importance,
    is here only for helping constructing the succint encoding of the 
    tree of nodes
    '''

@dataclass(frozen=True) # frozen=True is used to make the class immutable
class _AlgorithmT:
    """Represent the the type of split finding algorithm employed by the RSTree."""
    
    @classmethod
    def from_str(cls, name: str, epsilon: Optional[float] = None) -> "_AlgorithmT":
        name = name.lower().strip()
        if name == "exact":
            return _Exact()
        elif name == "approx":
            if epsilon is None:
                raise ValueError("An 'epsilon' value is required for the approximate algorithm.")
            return _Approximate(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown algorithm type: {name}")

@dataclass(frozen=True) 
class _Exact(_AlgorithmT):
    """
    Exact algorithm that enumerates over all possible splits
    on all the features.
    """

@dataclass(frozen=True)
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
        gradients: 
        hessians: 
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
        lmbda: a constant hyperparameter.
        gamma: a constant hyperparameter.
        algorithm: indicates the type of split algorithm used for constructing the tree.
        ranks: implements the rank query in how to move inside the succint encoding
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
        self.ranks: list[int] = []

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
        children : list[_ChildrenT] = []
        self._build_tree(data, 0, children)
        self._serialize(children)

    def _build_tree(self,
                    data: _Statistics,
                    depth: int,
                    children: list[_ChildrenT]):
        '''
        Constructs recursively the Boost Tree.

        Parameters:
            depth: the depth of the current node being constructed. If it is over the max_depth, it will construct a leaf instead and terminate the visit.
            data: the set of datas used to construct the node.

        Returns:
            the index of the node in the array
        '''
        # If we have reached the max_depth then construct a leaf,
        # otherwise try to construct an internal node
        if depth != self.max_depth:
            
            # CASE: construct an internal node
            split = None
            match self.algorithm:
                case _Exact():
                    split = self._split_exact(data)
                case _Approximate(epsilon = epsilon):
                    split = self._split_approx(data, epsilon)
                case _:
                    raise ValueError("Unknown Split Algorithm variant")
            
            # If the split is None then construct a leaf,
            # otherwise continue
            if split is not None:
                node = _Internal(threshold = split)
                
                self.nodes.append(node)
                children.append(_ChildrenT())
                
                node_idx = len(self.nodes) - 1 
                
                # Computing children
                left_child_idx = self._build_tree(
                    _Statistics(data.inputs[data.inputs[:, node.threshold.feature] <= node.threshold.value],
                                data.targets[data.inputs[:, node.threshold.feature] <= node.threshold.value],
                                data.gradients[data.inputs[:, node.threshold.feature] <= node.threshold.value],
                                data.hessians[data.inputs[:, node.threshold.feature] <= node.threshold.value]
                                ),
                    depth+1,
                    children)
                
                right_child_idx = self._build_tree(
                    _Statistics(data.inputs[data.inputs[:, node.threshold.feature] > node.threshold.value],
                                data.targets[data.inputs[:, node.threshold.feature] > node.threshold.value],
                                data.gradients[data.inputs[:, node.threshold.feature] > node.threshold.value],
                                data.hessians[data.inputs[:, node.threshold.feature] > node.threshold.value]
                                ),
                    depth+1,
                    children)
                
                # Add children
                children[node_idx].left = left_child_idx
                children[node_idx].right = right_child_idx

                return node_idx
            

        # CASE: construct a leaf
        pred = np.sum(data.gradients)/(np.sum(data.hessians) + self.lmbda)
        leaf = _Leaf(prediction=pred)

        self.nodes.append(leaf)
        children.append(_ChildrenT())
        leaf_idx = len(self.nodes) - 1 
        
        # Add dummy children for the leaf (Expand Phase of Succint Encoding)
        dummy_left = _Dummy()
        self.nodes.append(dummy_left)
        children.append(None)
        children[leaf_idx].left = len(self.nodes) - 1

        dummy_right = _Dummy()
        self.nodes.append(dummy_right)
        children.append(None)
        children[leaf_idx].right = len(self.nodes) - 1
        
        # Return index
        return leaf_idx
    
    def _serialize(self, children: list[_ChildrenT]):
        '''
        Applies a BFS over the expanded tree in order to construct
        finalise the succint encoding of the tree
        '''
        # Initialise the process
        queue = deque([0]) # start the BFS from the root
        old_nodes = self.nodes
        self.nodes = []
        old_to_new = {} # maps old indices to new indices
        count = 0 # counts the number of real nodes (i.e. not of type _Dummy) encountered so far
        self.ranks = []

        # SERIALIZE PHASE of Succint Encoding
        while queue:
            idx = queue.popleft()
            if idx not in old_to_new:
                # add to the mapping
                old_to_new[idx] = len(self.nodes)
                node = old_nodes[idx]
                match node:
                    case _Internal():
                        self.nodes.append(node)
                        count+=1
                    case _Leaf():
                        self.nodes.append(node)
                        count+=1
                self.ranks.append(count)

                # enqueue children if they exist
                match node:
                    case _Internal():
                        queue.append(children[idx].left)
                        queue.append(children[idx].right)
                    case _Leaf():
                        queue.append(children[idx].left)
                        queue.append(children[idx].right)
                      
    
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
                 np.array([self._r_minus(data.inputs[:,i], data.hessians, x) for x in feat_vals]),
                 np.array([self._r_plus(data.inputs[:, i], data.hessians, x) for x in feat_vals]),
                 np.array([self._omega(data.inputs[:, i], data.hessians, x) for x in feat_vals])
                 ) 

            # epsilon is the "approximation factor", s.t. the difference in rank
            # between two splits is less than epsilon
            start, end = np.min(S[3]), np.max(S[3])
            splits = []
            for d in np.linspace(start, end, int(1/epsilon)):
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
        if d < 0:
            raise ValueError("d must be greater than 0.")

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
        '''
        Return the similarity score of the data.
        '''
        return np.sum(gradients)**2 / (np.sum(hessians) + self.lmbda)
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        '''
        Predicts the output of each sample of input. 
        '''        
        pred = np.zeros(input.shape[0])
        
        for i in range(input.shape[0]):
            curr_idx = 0
            while True:
                try: 
                    match self.nodes[curr_idx]:
                        case _Internal(threshold):
                            if (input[i][threshold.feature] <= threshold.value):
                                # move left
                                l = ((curr_idx + 1) * 2) - 1
                                curr_idx = self.ranks[l] - 1
                            else:
                                # move right
                                r = ((curr_idx + 1) * 2)
                                curr_idx = self.ranks[r] - 1
                        case _Leaf(prediction):
                            pred[i] = prediction
                            break
                        case _Dummy():
                            raise TypeError('Error! Encountered a _Dummy node, only _Internal and _Leaf nodes should be present in a finalised tree.')
                except TypeError as error:
                    raise

        return pred





                
        





    