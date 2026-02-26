from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np

class Features(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_features(self):
        raise NotImplementedError('Implement make features')
    

class VolatilityFeatures(Features):
    def __init__(self):
        pass 

    def make_target(self, data: pd.DataFrame, horizon: int = 3_600):      
        if 'log_returns' not in data.columns: 
            data['log_returns'] = np.log(data['price']).diff()
        
        # Next hour price
        #data['target'] = np.log(data['price'].shift(-horizon) / data['price'])
        # Next hour volatility
        data['target'] = data['log_returns'].pow(2).rolling(window = horizon).sum().shift(-horizon).pow(0.5)

        return data.dropna()
    
    def make_lagged_prices(self,data: pd.DataFrame, lag: int = 4, horizon: int = 3_600):
        

        for i in range(1, lag+1):
            data[f'price_lag_{i}_hour'] = data['price'].shift(horizon*i)
        return data.dropna()
    
    def make_lagged_volatility(self, data: pd.DataFrame, lag: int = 4, horizon: int = 3_600):
        if 'log_returns' not in data.columns: 
            data['log_returns'] = np.log(data['price']).diff()


        for i in range(1, lag+1):
            data[f'volatility_lag_{i}_hour'] = data['log_returns'].pow(2).rolling(window = horizon*i).sum().pow(0.5)
        
        return data.dropna()
        
    def make_lagged_log_returns(self, data: pd.DataFrame, lag: int = 4, horizon: int = 3_600):
        if 'log_returns' not in data.columns: 
            data['log_returns'] = np.log(data['price']).diff()

        for i in range(1, lag+1):
            data[f'log_return_lag_{i}_hour'] = data['log_returns'].shift(horizon*i)
        return data.dropna()

    def make_features(self, data: pd.DataFrame, lag: int = 4, horizon: int = 3_600):
        data = data.copy()

        data = self.make_target(data, horizon)
        data = self.make_lagged_prices(data, lag, horizon)
        data = self.make_lagged_volatility(data, lag, horizon)
        data = self.make_lagged_log_returns(data, lag, horizon)

        return data 

        