

class Preprocess():
    import numpy as np
    def __init__(self, transformations:list) -> None:
        self.transformations = []
        for name in transformations:
            if name == "MinMaxScaler":
                from sklearn.preprocessing import MinMaxScaler
                self.transformations.append(MinMaxScaler())
            elif name == "StandardScaler":
                from sklearn.preprocessing import StandardScaler
                self.transformations.append(StandardScaler())



    def transform(self, X):
        for transformation in self.transformations:
            X = transformation.transform(X)
        return X

    def fit_transform(self, X):
        for transformation in self.transformations:
            X = transformation.fit_transform(X)
        return X

    def inverse_transform(self, X):
        for transformation in self.transformations:
            X = transformation.inverse_transform(X)
        return X
    
    def transform_dataset(self, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray):
        for transformation in self.transformations:
            X_train = transformation.fit_transform(X_train)
            X_test = transformation.transform(X_test)
        
        return X_train, X_test

        
