import pandas as pd
import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass

class MyTreeReg:
    
    @dataclass 
    class Container:
        type: str
        ptr: Union['MyTreeReg.Node', 'MyTreeReg.Leaf']
        
    @dataclass
    class Node:
        sep: float = None
        column: Union[str, int] = None
        left: Union['MyTreeReg.Node', 'MyTreeReg.Leaf'] = None
        right: Union['MyTreeReg.Node', 'MyTreeReg.Leaf'] = None
        
    @dataclass 
    class Leaf:
        value: np.ndarray
    
    @dataclass
    class BestSplit:
        column: Union[str, int] = None
        sep: float = None
        mse_gain: float = None
        left: pd.Series = None
        right: pd.Series = None
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20) -> None:
        self._max_depth = max_depth
        self._depth = 0
        self._min_samples_split = min_samples_split
        self._max_leafs = max_leafs if max_leafs > 1 else 2
        self._leafs_cnt = 0
        self._expanded = 0
        self._tree = None
    
    def __str__(self) -> str:
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"        
        
    def _calc_mse(self, y: pd.Series) -> float:
        return np.mean((y - np.mean(y))**2)
    
    def _calc_mse_gain(self, labels: pd.Series, left: pd.Series, right: pd.Series) -> float:
        mse_origin = self._calc_mse(labels)
        mse_left = self._calc_mse(left)
        mse_right = self._calc_mse(right)

        return mse_origin - left.size / labels.size * mse_left - right.size / labels.size * mse_right
        
    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Union[str, int], float, float]:
        best_split: 'MyTreeReg.BestSplit' = self.BestSplit()
        
        for column in X.columns:
            attrs: pd.Series = X[column].sort_values()
            unqiue_values: np.ndarray = attrs.unique()
            separators = np.array([(unqiue_values[i] + unqiue_values[i + 1]) / 2 for i in range(unqiue_values.size - 1)])

            for sep in separators:
                left, right = y[attrs <= sep], y[attrs > sep]
                mse_gain = self._calc_mse_gain(y, left, right)
                
                if best_split.mse_gain is None:
                    best_split.mse_gain = mse_gain
                    best_split.column = column
                    best_split.sep = sep
                    best_split.left = left
                    best_split.right = right
                else:
                    if best_split.mse_gain < mse_gain:
                        best_split.mse_gain = mse_gain
                        best_split.column = column
                        best_split.sep = sep
                        best_split.left = left
                        best_split.right = right
            
        return best_split

    def _is_leaf(self, y: pd.Series) -> bool:
        return y.size < self._min_samples_split or self._calc_mse(y) == 0.0
    
    def _is_expandable(self) -> bool:
        return self._expanded + 2 <= self._max_leafs
    
    def _create_leaf(self, y: pd.Series) -> 'MyTreeReg.Containter':
        self._leafs_cnt += 1
        return self.Container(type='leaf', ptr=self.Leaf(value=y))
    
    def _fit(self, X: pd.DataFrame, y: pd.Series) -> 'MyTreeReg.Container':
        if self._depth < self._max_depth:
            if not self._is_leaf(y) and self._is_expandable():
                best_split: 'MyTreeReg.BestSplit' = self.BestSplit()
                best_split = self.get_best_split(X, y)
                
                labels_left, labels_right = best_split.left, best_split.right
                features_left, features_right = X.loc[labels_left.index], X.loc[labels_right.index]
                cnt = self.Container(type='node', ptr=self.Node(column=best_split.column, sep=best_split.sep))
                self._expanded += 1
                    
                self._depth += 1
                cnt.ptr.left = self._fit(features_left, labels_left)
                cnt.ptr.right = self._fit(features_right, labels_right)
                self._depth -= 1
                    
                return cnt
            else:
                return self._create_leaf(y)
        else:
            return self._create_leaf(y)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._tree = self._fit(X, y)
        
    def _build_tree(self, cntr: Union['MyTreeReg.Node', 'MyTreeReg.Leaf'], level: int, leafs_sum: float) -> None:
        if cntr.type == 'node':
            column = cntr.ptr.column if isinstance(cntr.ptr.column, str) else str(cntr.ptr.column)
            print((level - 1) * '\t' + column + ': ')
            print((level - 1) * '\t', end='')
            print(cntr.ptr.sep)
            print((level - 1) * '\t' + 'left:')
            leafs_sum += self._build_tree(cntr.ptr.left, level+1, 0)
            print((level - 1) * '\t' + 'right:')
            leafs_sum += self._build_tree(cntr.ptr.right, level+1, 0)
            
            return leafs_sum
        
        if cntr.type == 'leaf':
            print(level * '\t' + 'sum: ')
            print(level * '\t', end='')
            leaf_sum = np.mean(cntr.ptr.value)
            print(leaf_sum)
            
            return leaf_sum
            
    def build_tree(self) -> float:
        leafs_sum = 0.0
        return self._build_tree(self._tree, 1, leafs_sum)
    
    def _predict(self, X: pd.DataFrame, cntr: Union['MyTreeReg.Node', 'MyTreeReg.Leaf']) -> float:
        if cntr.type == 'leaf':
            return np.mean(cntr.ptr.value)
        
        if cntr.type == 'node':
            if X[cntr.ptr.column] <= cntr.ptr.sep and cntr.ptr.left is not None:
                return self._predict(X, cntr.ptr.left)
            if X[cntr.ptr.column] > cntr.ptr.sep and cntr.ptr.right is not None:
                return self._predict(X, cntr.ptr.right)
                    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dataset = X.reset_index(drop=True)
        values = np.zeros(dataset.shape[0], dtype='float')
        
        for index, row in dataset.iterrows():
            values[index] = self._predict(row, self._tree)
        
        return values