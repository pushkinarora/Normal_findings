from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ResidualTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        y_preds = self.model.predict(X)
        return (y - y_preds).reshape(-1, 1)  # Return residuals as a 2D array for compatibility
