import pandas as pd
import numpy as np
from typing import Tuple
from dataclasses import dataclass

class MyTreeClf:
    
    @dataclass
    class BestSplit:
        ig: int = 0
        sep: float = 0.0
        column: str = ''
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self._best_split = self.BestSplit()
        
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
    
    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[str, float, float]:
        for column in X.columns:
            attr: pd.Series = X[column].sort_values()
            unqiue_values: np.ndarray = attr.unique()
            separators = np.array([(unqiue_values[i] + unqiue_values[i + 1]) / 2 for i in range(unqiue_values.size - 1)])
            origin_entropy = self.calc_entropy(y)

            for sep in separators:
                left, right = y[attr <= sep], y[attr > sep]
                ig = origin_entropy - (left.size / y.size * self.calc_entropy(left) + right.size / y.size * self.calc_entropy(right))
                
                if ig > self._best_split.ig:
                    self._best_split.ig = ig
                    self._best_split.sep = sep
                    self._best_split.column = str(column)
                    
        return self._best_split.column, self._best_split.sep, self._best_split.ig