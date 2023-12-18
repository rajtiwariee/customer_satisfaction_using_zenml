import logging
from abc import ABC, abstractmethod
from logs.customlog import logger
# import optuna
import pandas as pd
# import xgboost as xgb
# from lightgbm import LGBMRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """
    
    @abstractmethod
    def train(self, x_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

    # @abstractmethod
    # def optimize(self, trial, x_train, y_train, x_test, y_test):
    #     """
    #     Optimizes the hyperparameters of the model.

    #     Args:
    #         trial: Optuna trial object
    #         x_train: Training data
    #         y_train: Target data
    #         x_test: Testing data
    #         y_test: Testing target
    #     """
    #     pass
    
    
class LinearRegressionModel(Model):
    
    """
    Linear Regression Model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        
        Returns : 
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logger.info("Model Training completed")
            return reg
        except Exception as e:
            logger.error("Training unsuccessful")
            raise e
        
    
