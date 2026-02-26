from .machine_learning import LinearRegression, XGBoost, DecisionTree, ARMA, AR1, ARIMA, GARCH, Persistence

def create_model(algorithm_name:str, hyperparameters:dict, seed:int = 1337):
    model = None
    if algorithm_name == "Linear Regression":
        model = LinearRegression(hyperparameters)
    elif algorithm_name == "Decision Tree":
        hyperparameters["random_state"] = seed
        model = DecisionTree(hyperparameters)
    elif algorithm_name == "XGBoost": 
        hyperparameters["random_state"] = seed
        model = XGBoost(hyperparameters)
    elif algorithm_name == "ARMA":
        hyperparameters["random_state"] = seed
        model = ARMA(hyperparameters)
    elif algorithm_name == "AR1":
        hyperparameters["random_state"] = seed
        model = AR1(hyperparameters)
    elif algorithm_name == "ARIMA":
        hyperparameters["random_state"] = seed
        model = ARIMA(hyperparameters)
    elif algorithm_name == "GARCH":
        model = GARCH(hyperparameters)
    elif algorithm_name == "Persistent":
        model = Persistence(hyperparameters)
    return model
    



# For more information, see: https://github.com/scikit-learn/scikit-learn/blob/872124551/sklearn/model_selection/_search.py#L61
from typing import Mapping
import itertools
def generate_grid(hyperpameters):
    if isinstance(hyperpameters, Mapping):
        # dict of lists
        hyperparameters_grid = [hyperpameters]
    else:
        # list of dicts of lists
        hyperparameters_grid = hyperpameters

    for hyperpameters_dict in hyperparameters_grid:
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(hyperpameters_dict.items())
        if not items:
            yield {}
        else:
            # Get the keys and the values from zipping the dict with itself.
            keys, values = zip(*items)
            # For every value, do the cross product with the other values.
            for v in itertools.product(*values):
                # Combines the keys with a single cross-product result (v-tuple)
                params = dict(zip(keys, v))
                # yiled the dictionary formed with a single v-tuple
                yield params