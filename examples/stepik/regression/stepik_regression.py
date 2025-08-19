from typing import Dict
from sklearn.datasets import load_diabetes

from decision_tree.decision_tree_reg import MyTreeReg

def load_dataset():
    return load_diabetes(return_X_y=True, as_frame=True)
    

def stepik_regression(example_config: Dict[str, int]):
    X, y = load_dataset()
    
    model = MyTreeReg(**example_config)
    model.fit(X, y)
    print(model._leafs_cnt)
    model.build_tree()