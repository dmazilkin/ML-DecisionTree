from typing import List

from examples.stepik.regression.stepik_regression import load_dataset
from src.decision_tree_reg import MyTreeReg
from helpers.config_parser import read_config

REG_CONFIGS_PATH = r'examples/stepik/regression/'

CONFIGS: List[str]  = [
    r'configs/tree1.config', 
    r'configs/tree2.config', 
    r'configs/tree3.config', 
    r'configs/tree4.config', 
    r'configs/tree5.config', 
    r'configs/tree6.config',
]

LEAF_CNT: List[int] = [
    2,
    5,
    7,
    11,
    21,
    27,
]

def test_stepik_regression_leaf_cnt():
    X, y = load_dataset()
    
    for ind, config_path in enumerate(CONFIGS):
        config = read_config(REG_CONFIGS_PATH + config_path)
        model = MyTreeReg(**config)
        model.fit(X, y)
        assert model._leafs_cnt == LEAF_CNT[ind]