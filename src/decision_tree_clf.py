import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass

class MyTreeClf:
    
    @dataclass
    class BestSplit:
        left: np.ndarray = None
        right: np.ndarray = None
        ig: int = 0
        sep: float = 0.0
        column: str = ''
        
    @dataclass
    class Node:
        sep: float = None
        column: str = None
        left: Union['Node', np.ndarray] = None
        right: Union['Node', np.ndarray] = None
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs if max_leafs > 1 else 2
        self.leafs_cnt = 0
        self._depth = 1
        self.tree = None
        
    def __str__(self) -> str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"
    
    def calc_entropy(self, y: pd.Series) -> float:
        entropy = 0
        labels_count = y.size
            
        for label in y.unique():
            proba = np.sum(y == label) / labels_count
            
            if proba != 0:
                entropy += -1 * proba * np.log2(proba)
            
        return entropy
    
    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float, float, pd.DataFrame, pd.DataFrame]:
        best_split = self.BestSplit()
        
        for column in X.columns:
            attr: pd.Series = X[column].sort_values()
            unqiue_values: np.ndarray = attr.unique()
            separators = np.array([(unqiue_values[i] + unqiue_values[i + 1]) / 2 for i in range(unqiue_values.size - 1)])
            origin_entropy = self.calc_entropy(y)

            for sep in separators:
                left, right = y[attr <= sep], y[attr > sep]
                ig = origin_entropy - (left.size / y.size * self.calc_entropy(left) + right.size / y.size * self.calc_entropy(right))
                
                if ig > best_split.ig:
                    best_split.ig = ig
                    best_split.sep = sep
                    best_split.column = str(column)
                    best_split.left = left
                    best_split.right = right
                    
        return best_split.column, best_split.sep, best_split.ig, best_split.left, best_split.right
    
    def _is_leaf(self, y: pd. Series) -> bool:
        return y.size < self.min_samples_split or self.calc_entropy(y) == 0.0
    
    def _build_tree(self, node: Node, level: int) -> None:
        if not isinstance(node, pd.Series):
            print((level - 1) * '\t' + node.column + ': ')
            print((level - 1) * '\t', end='')
            print(node.sep)
            print((level - 1) * '\t' + 'left:')
            self._build_tree(node.left, level+1)
            if node.right is not None:
                print((level - 1) * '\t' + 'right:')
                self._build_tree(node.right, level+1)
        else:
            print(level * '\t' + 'proba: ')
            print(level * '\t', end='')
            print(np.sum(node == 1) / node.size)
        
    def build_tree(self) -> None:
        self._build_tree(self.tree, 1)
    
    def _fit(self, X: pd.DataFrame, y: pd.Series, parent: Node) -> Union[Node, pd.Series]:
        if (self.max_depth >= self._depth):
            if not self._is_leaf(y) and (self.leafs_cnt + 2 <= self.max_leafs):
                column, sep, ig, left, right = self.get_best_split(X, y)
                node = self.Node(sep=sep, column=column)
                            
                self._depth += 1
                node.left = self._fit(X.loc[left.index], left, node)
                node.right = self._fit(X.loc[right.index], right, node)
                self._depth -= 1
                
                if node.left is None and node.right is None:
                    if self.leafs_cnt < self.max_leafs:                
                        self.leafs_cnt += 1
                        return y
                    else:
                        return None
                else:
                    return node
            else:
                if self.leafs_cnt < self.max_leafs:
                    self.leafs_cnt += 1
                
                    return y
                else:
                    return None
        else:
            if self.leafs_cnt < self.max_leafs:
                self.leafs_cnt += 1
                
                return y
            else:
                return None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.tree = self._fit(X, y, parent=None)
            