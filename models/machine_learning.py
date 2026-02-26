class MachineLearning():
    def __init__(self, hyperparameters:dict):
        self.hyperparameters = hyperparameters

    def train(self, X, y):
        raise NotImplementedError("Subclass of Model should implement train")
    
    def __call__(self, X):
        raise NotImplementedError("Subclass of Model should implement __call__")
    
    def reset(self):
        raise NotImplementedError("Subclass of Model should implement reset")

class DecisionTree(MachineLearning):
    from sklearn.tree import DecisionTreeRegressor
    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        
        self.model = self.DecisionTreeRegressor(**self.hyperparameters)

    def train(self, X, y):
        self.model.fit(X, y)

    def __call__(self, X):
        return self.model.predict(X).reshape((-1,1))
    
    def reset(self):
        self.model = self.DecisionTreeRegressor(**self.hyperparameters)
    
class LinearRegression(MachineLearning):
    from sklearn.linear_model import LinearRegression
    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        

        self.model = self.LinearRegression(**self.hyperparameters)  

    def train(self, X, y):
        self.model.fit(X, y)

    def __call__(self, X):
        return self.model.predict(X).reshape((-1,1))
       
    
    def reset(self):
        self.model = self.LinearRegression(**self.hyperparameters)

    
class XGBoost(MachineLearning):
    from xgboost import XGBRegressor
    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.model = self.XGBRegressor(**self.hyperparameters) 

    def train(self, X, y):
        self.model.fit(X, y)
    
    def __call__(self, X):
        return self.model.predict(X).reshape((-1,1))
    
    def reset(self):
        self.model = self.XGBRegressor(**self.hyperparameters)

class ARMA(MachineLearning):
    from statsmodels.tsa.arima.model import ARIMA

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.model = None

    def train(self, X, y):
        # Expect hyperparameters to contain:
        # "p" and "q"
        p = self.hyperparameters.get("p", 1)
        q = self.hyperparameters.get("q", 1)

        self.model = self.ARIMA(
            y,
            order=(p, 0, q)
        ).fit()

    def __call__(self, X):
        return self.model.forecast(steps=len(X)).reshape((-1,1))

    def reset(self):
        self.model = None

class AR1(MachineLearning):
    from statsmodels.tsa.arima.model import ARIMA
    import numpy as np

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.model = None
        self.fitted_model = None

    def train(self, X, y):   # <- match interface
        y = y.reshape(-1)
        self.model = self.ARIMA(
            y,
            order=(1, 0, 0),
            trend=self.hyperparameters.get("trend", "c"),
            enforce_stationarity=self.hyperparameters.get("enforce_stationarity", True),
            enforce_invertibility=self.hyperparameters.get("enforce_invertibility", True)
        )
        self.fitted_model = self.model.fit()

    def __call__(self, X):
        steps = len(X)
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.reshape(-1, 1)

    def reset(self):
        self.model = None
        self.fitted_model = None

class ARIMA(MachineLearning):
    from statsmodels.tsa.arima.model import ARIMA

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.model = None
        self.fitted_model = None

    def train(self, X, y):   # <- match interface
        y = y.reshape(-1)
        p = self.hyperparameters.get("p", 1)
        d = self.hyperparameters.get("d", 0)
        q = self.hyperparameters.get("q", 0)

        self.model = self.ARIMA(
            y,
            order=(p, d, q),
            trend=self.hyperparameters.get("trend", "c"),
            enforce_stationarity=self.hyperparameters.get("enforce_stationarity", True),
            enforce_invertibility=self.hyperparameters.get("enforce_invertibility", True)
        )
        self.fitted_model = self.model.fit()

    def __call__(self, X):
        steps = len(X)
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.reshape(-1, 1)

    def reset(self):
        self.model = None
        self.fitted_model = None

from arch.univariate import arch_model
class GARCH(MachineLearning):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        self.model = None
        self.fitted_model = None

    def train(self, X, y):   # <- match interface
        y = y.reshape(-1)
        
        self.model = arch_model(
            y,
            **self.hyperparameters
        )
        self.fitted_model = self.model.fit(disp="off")

    def __call__(self, X):
        steps = len(X)
        forecast = self.fitted_model.forecast(horizon=steps)
        variance = forecast.variance.values[-1, :]
        return variance.reshape(-1, 1)

    def reset(self):
        self.model = None
        self.fitted_model = None

class Persistence(MachineLearning):
    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

    def train(self, X, y):
        pass
    
    def __call__(self, X):
        return X[:, self.hyperparameters["column_number"]].reshape((-1,1))
    
    def reset(self):
        pass