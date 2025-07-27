import pandas as pd
from matplotlib import pyplot as plt
from typing import Dict, Tuple
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.decision_tree_clf import MyTreeClf

def generate_data_classification() -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, random_state=42)
    
    return pd.DataFrame(X), pd.Series(y)

def classification(example_config: Dict[str, int]):
    X, y = generate_data_classification()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
    model = MyTreeClf(**example_config)
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    model.build_tree()
      
    figure, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].scatter(X[y == 1][0], X[y == 1][1], color='r')
    axes[0].scatter(X[y == 0][0], X[y == 0][1], color='b')
    axes[0].set_title('Training dataset')
    axes[1].scatter(X_test[y_predict == 1][0], X_test[y_predict == 1][1], color='r')
    axes[1].scatter(X_test[y_predict == 0][0], X_test[y_predict == 0][1], color='b')
    axes[1].set_title('Predicted Test dataset')
    axes[2].scatter(X_test[y_test == 1][0], X_test[y_test == 1][1], color='r')
    axes[2].scatter(X_test[y_test == 0][0], X_test[y_test == 0][1], color='b')
    axes[2].set_title('Real Test dataset')
    plt.show()