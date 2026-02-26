import pandas as pd

def get_dataset_split(cfg:dict, X:pd.DataFrame, y:pd.DataFrame):
    if cfg["SPLIT_TYPE"] == "Random":
        return RandomFold(X, y, n_splits=5, shuffle=True, seed=cfg["SEED"]).split()
    elif cfg["SPLIT_TYPE"] == "Stratified":
        return StratifiedFold(X, y, n_splits=5, num_quantiles=10, shuffle=True, seed=cfg["SEED"]).split()
    elif cfg["SPLIT_TYPE"] == "TimeSeries":
        return TimeSeriesFold(X, y, n_splits=10).split()
    return None

class CrossValidation():
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame) -> None:
        self.X = X
        self.y = y

    
    def split(self):
        raise NotImplementedError("Subclass of Split should implement split")
    
class RandomFold(CrossValidation):
    from sklearn import model_selection 
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, n_splits:int = 5, shuffle:bool = True, seed:int = 1337) -> None:
        super().__init__(X, y)

        self.cross_validator = self.model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    def split(self):
        generator = self.cross_validator.split(self.X, self.y) 
        train_indices, test_indices = zip(*list(generator))
        return train_indices, test_indices
    

class StratifiedFold(CrossValidation):
    from sklearn import model_selection
    def __init__(self, X:pd.DataFrame, y:pd.DataFrame, n_splits:int = 5, num_quantiles:int = 10,
            shuffle:bool = True, seed:int = 1337) -> None:
        self.X = X

        labels = [str(q) for q in range(num_quantiles)]
        self.discrete_y = pd.qcut(y.T.squeeze(axis=0).rank(method="first"), q=num_quantiles, labels=labels) 

        self.cross_validator = self.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    def split(self):
        generator = self.cross_validator.split(self.X, self.discrete_y)
        train_indices, test_indices = zip(*list(generator))
        return train_indices, test_indices

class TimeSeriesFold(CrossValidation):
    from sklearn import model_selection
    def __init__(self, X, y, n_splits:int = 5):
        super().__init__(X, y)
        
        self.timeseries_validator = self.model_selection.TimeSeriesSplit(n_splits=n_splits)

    def split(self):
        generator = self.timeseries_validator.split(self.X, self.y)
        train_indices, test_indices = zip(*list(generator))

        return train_indices, test_indices