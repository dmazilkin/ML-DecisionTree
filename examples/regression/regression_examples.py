import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, Tuple
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.decision_tree_reg import MyTreeReg

def generate_data_regression() -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(n_samples=100, n_features=1, n_informative=1, noise=5, random_state=42)
    
    return pd.DataFrame(X), pd.Series(y)

def regression():
    X, y = generate_data_regression()
    
    model = MyTreeReg()
    column, sep, gain = model.get_best_split(X, y)
    print(column, sep, gain)
        
    # figure, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].scatter(X, y, color='r')
    # axes[0].set_title('Training dataset')
    # plt.show()