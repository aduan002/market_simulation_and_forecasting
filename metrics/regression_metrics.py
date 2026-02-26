

class RegressionMetric():
    import sklearn.metrics as sk_metrics
    import statistics as stats
    def __init__(self) -> None:
        self.scores = {
            "mean_absolute_error": [],
            "mean_squared_error": [],
            "max_error": [], # calculates the maximum residual error 
            "r2": []
        }

    def update(self, y, y_hat):
        self.scores["mean_absolute_error"].append(self.sk_metrics.mean_absolute_error(y, y_hat))
        self.scores["mean_squared_error"].append(self.sk_metrics.mean_squared_error(y, y_hat))
        self.scores["max_error"].append(self.sk_metrics.max_error(y, y_hat))
        self.scores["r2"].append(self.sk_metrics.r2_score(y, y_hat))

    def mean(self):
        mean_metrics = {}
        for metric, values in self.scores.items():
            mean_metrics[metric] = self.stats.mean(values)
        return mean_metrics
    
    def stdev(self):
        std_metrics = {}
        for metric, values in self.scores.items():
            std_metrics[metric] = self.stats.pstdev(values)
        return std_metrics
    
    def reset(self):
        self.scores = {
            "mean_absolute_error": [],
            "mean_squared_error": [],
            "max_error": [],
            "r2": []
        }

    def __call__(self, y, y_hat):
        scores = {
            "mean_absolute_error": self.sk_metrics.mean_absolute_error(y, y_hat),
            "mean_squared_error": self.sk_metrics.mean_squared_error(y, y_hat),
            "max_error": self.sk_metrics.max_error(y, y_hat), # calculates the maximum residual error 
            "r2": self.sk_metrics.r2_score(y, y_hat)
        }

        return scores
