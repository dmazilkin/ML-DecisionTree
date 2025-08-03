import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, Tuple
from sklearn.datasets import make_classification

from src.decision_tree_clf import MyTreeClf

DATA_PATH = r'examples/stepik/classification/data_banknote_authentication.txt'

def read_data_classification() -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
    X, y = df.iloc[:,:4], df['target']
    
    return X, y

def stepik_classification(example_config: Dict[str, int]):
    X, y = read_data_classification()
    
    model = MyTreeClf(**example_config)
    model.fit(X, y)
    
    print(model.leafs_cnt)
    leafs_sum = model.build_tree()
    print(leafs_sum)
    print(model._fi)