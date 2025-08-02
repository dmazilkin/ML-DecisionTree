import pandas as pd
import numpy as np
from typing import Tuple, Union
from dataclasses import dataclass

class MyTreeReg:
    
    @dataclass
    class BestSplit:
        column: Union[str, int] = None
        sep: float = None
        mse_gain: float = None
    
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20) -> None:
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._max_leafs = max_leafs
    
    def __str__(self) -> str:
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"        
        
    def _calc_mse_gain(self, labels: pd.Series, left: pd.Series, right: pd.Series) -> float:
        origin = np.mean((labels - np.mean(labels))**2)
        mse_left = np.mean((left - np.mean(left))**2)
        mse_right = np.mean((right - np.mean(right))**2)
        print(origin, mse_left, mse_right, origin - left.size / labels.size * mse_left - right.size / labels.size * mse_right)
        return origin - left.size / labels.size * mse_left - right.size / labels.size * mse_right
        
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
                else:
                    if best_split.mse_gain < mse_gain:
                        best_split.mse_gain = mse_gain
                        best_split.column = column
                        best_split.sep = sep
            
        return best_split.column, best_split.sep, best_split.mse_gain