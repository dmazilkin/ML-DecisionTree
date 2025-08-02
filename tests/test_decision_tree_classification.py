from typing import List

from examples.stepik.stepik_classification import read_data_classification
from src.decision_tree_clf import MyTreeClf
from helpers.config_parser import read_config

CONFIGS: List[str]  = [
    r'examples/stepik/configs/tree1.config', 
    r'examples/stepik/configs/tree2.config', 
    r'examples/stepik/configs/tree3.config', 
    r'examples/stepik/configs/tree4.config', 
    r'examples/stepik/configs/tree5.config', 
    r'examples/stepik/configs/tree6.config',
]

CONFIGS_BINS: List[str]  = [
    r'examples/stepik/configs_bins/tree1.config', 
    r'examples/stepik/configs_bins/tree2.config', 
    r'examples/stepik/configs_bins/tree3.config', 
    r'examples/stepik/configs_bins/tree4.config', 
    r'examples/stepik/configs_bins/tree5.config', 
    r'examples/stepik/configs_bins/tree6.config',
]

CONFIGS_CRITERION: List[str]  = [
    r'examples/stepik/configs_criterion/tree1.config', 
    r'examples/stepik/configs_criterion/tree2.config', 
    r'examples/stepik/configs_criterion/tree3.config', 
    r'examples/stepik/configs_criterion/tree4.config', 
    r'examples/stepik/configs_criterion/tree5.config', 
    r'examples/stepik/configs_criterion/tree6.config',
]

LEAF_CNT: List[int] = [
    2,
    5,
    9,
    12,
    18,
    21,
]

LEAF_SUM: List[int] = [
    0.918956,
    2.916956,
    4.796617,
    5.969142,
    6.604379,
    7.82549,
]

LEAF_SUM_BINS: List[int] = [
    0.71033,
    2.916956,
    5.020575,
    5.85783,
    9.526468,
    12.025427,
]

LEAF_CNT_CRITERION: List[int] = [
    2,
    5,
    10,
    11,
    21,
    27,
]

LEAF_SUM_CRITERION: List[int] = [
    0.981148,
    2.799994,
    5.020575,
    5.200813,
    10.198869,
    12.412269,
]

def test_stepik_classification_leafs_cnt():
    X, y = read_data_classification()
    
    for ind, config_path in enumerate(CONFIGS):
        config = read_config(config_path)
        model = MyTreeClf(**config)
        model.fit(X, y)
        assert model.leafs_cnt == LEAF_CNT[ind]

def test_stepik_classification_leafs_sum():
    X, y = read_data_classification()
    
    for ind, config_path in enumerate(CONFIGS):
        config = read_config(config_path)
        model = MyTreeClf(**config)
        model.fit(X, y)
        assert round(model.build_tree(), 6) == LEAF_SUM[ind]
        
def test_stepik_classification_leafs_sum_bins():
    X, y = read_data_classification()
    
    for ind, config_path in enumerate(CONFIGS_BINS):
        config = read_config(config_path)
        model = MyTreeClf(**config)
        model.fit(X, y)
        assert round(model.build_tree(), 6) == LEAF_SUM_BINS[ind]
        
def test_stepik_classification_criterion():
    X, y = read_data_classification()
    
    for ind, config_path in enumerate(CONFIGS_CRITERION):
        config = read_config(config_path)
        model = MyTreeClf(**config)
        model.fit(X, y)
        assert model.leafs_cnt == LEAF_CNT_CRITERION[ind]
        assert round(model.build_tree(), 6) == LEAF_SUM_CRITERION[ind]
