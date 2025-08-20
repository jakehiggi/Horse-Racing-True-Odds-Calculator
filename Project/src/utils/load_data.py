import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class DataLoader(BaseEstimator, TransformerMixin,
                 feature_engineering=True):
    def __init__(self):
        self.feature_engineering = feature_engineering
        

