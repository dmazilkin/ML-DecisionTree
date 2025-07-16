import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, Tuple
from sklearn.datasets import make_classification

from src.decision_tree_clf import MyTreeClf

def read_data_classification() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv('examples/stepik/data_banknote_authentication.txt', header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    
    return X, y

def stepik_classification(example_config: Dict[str, int]):
    X, y = read_data_classification()
    
    model = MyTreeClf(**example_config)
    model.fit(X, y)
    
    print(model.leafs_cnt)
    model.build_tree()