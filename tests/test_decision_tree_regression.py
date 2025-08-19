from typing import List

from examples.stepik.regression.stepik_regression import load_dataset
from decision_tree.decision_tree_reg import MyTreeReg
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

CONFIGS_BINS: List[str]  = [
    r'configs_bins/tree1.config', 
    r'configs_bins/tree2.config', 
    r'configs_bins/tree3.config', 
    r'configs_bins/tree4.config', 
    r'configs_bins/tree5.config', 
    r'configs_bins/tree6.config',
]

LEAF_CNT: List[int] = [
    2,
    5,
    7,
    11,
    21,
    27,
]

LEAF_SUM: List[float] = [
    303.138024,
    813.992098,
    1143.916064,
    1808.268095,
    3303.816014,
    4352.894213,
]

LEAF_CNT_BINS: List[int] = [
    2,
    5,
    9,
    12,
    21,
    30,
]

LEAF_SUM_BINS: List[float] = [
    324.685749,
    813.992098,
    1534.009043,
    2031.142507,
    3483.584468,
    4389.213407,
]

def test_stepik_regression_leaf_cnt():
    X, y = load_dataset()
    
    for ind, config_path in enumerate(CONFIGS):
        config = read_config(REG_CONFIGS_PATH + config_path)
        model = MyTreeReg(**config)
        model.fit(X, y)
        assert model._leafs_cnt == LEAF_CNT[ind]
        
def test_stepik_regression_leaf_sum():
    X, y = load_dataset()
    
    for ind, config_path in enumerate(CONFIGS):
        config = read_config(REG_CONFIGS_PATH + config_path)
        model = MyTreeReg(**config)
        model.fit(X, y)
        assert round(model.build_tree(), 6) == LEAF_SUM[ind]
        
def test_stepik_regression_leaf_cnt_bins():
    X, y = load_dataset()
    
    for ind, config_path in enumerate(CONFIGS_BINS):
        config = read_config(REG_CONFIGS_PATH + config_path)
        model = MyTreeReg(**config)
        model.fit(X, y)
        assert model._leafs_cnt == LEAF_CNT_BINS[ind]

def test_stepik_regression_leaf_sum_bins():
    X, y = load_dataset()
    
    for ind, config_path in enumerate(CONFIGS_BINS):
        config = read_config(REG_CONFIGS_PATH + config_path)
        model = MyTreeReg(**config)
        model.fit(X, y)
        assert round(model.build_tree(), 6) == LEAF_SUM_BINS[ind]