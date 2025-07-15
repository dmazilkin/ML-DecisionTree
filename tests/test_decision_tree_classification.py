import pytest
from typing import List

from examples.stepik.stepik_classification import read_data_classification
from src.decision_tree_clf import MyTreeClf
from helpers.config_parser import read_config

CONFIGS: List[str]  = [
    r'examples/stepik/tree1.config', 
    r'examples/stepik/tree2.config', 
    r'examples/stepik/tree3.config', 
    r'examples/stepik/tree4.config', 
    r'examples/stepik/tree5.config', 
    r'examples/stepik/tree6.config',
]

LEAF_CNT: List[int] = [
    2,
    5,
    9,
    12,
    18,
    21,
]

def test_stepik_classification():
    X, y = read_data_classification()
    
    for ind, config_path in enumerate(CONFIGS):
        config = read_config(config_path)
        model = MyTreeClf(**config)
        model.fit(X, y)
        assert model.leafs_cnt == LEAF_CNT[ind]



