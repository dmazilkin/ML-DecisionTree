import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, Tuple
from sklearn.datasets import make_classification

from src.decision_tree_clf import MyTreeClf

def generate_data_classification() -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, random_state=42)
    
    return pd.DataFrame(X), pd.Series(y)

def classification(example_config: Dict[str, int]):
    X, y = generate_data_classification()
    
    model = MyTreeClf(**example_config)
    model.fit(X, y)
    
    print(model.leafs_cnt)
    model.build_tree()
      
    figure, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].scatter(X[y == 1][0], X[y == 1][1], color='r')
    axes[0].scatter(X[y == 0][0], X[y == 0][1], color='b')
    axes[0].set_title('Original dataset')
    
    axes[1].set_title('Predicted dataset')
    plt.show()