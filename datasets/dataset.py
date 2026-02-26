import pandas as pd 


class TabularDataset:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.y = y.to_numpy() if isinstance(y, pd.DataFrame) else y

    def __len__(self): 
        return self.X.shape[0]

    def __getitem__(self, idx): 
       return self.X[idx], self.y[idx]
        


    
