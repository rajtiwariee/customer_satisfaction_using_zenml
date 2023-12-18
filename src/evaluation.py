from logs.customlog import logger
from abc import ABC, abstractmethod
import numpy as np 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

class Evaluation(ABC):
    
    """
    Abstract class defining strategy for evaluation of our models
    """
    @abstractmethod
    def calculate_scores(self, y_true : np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the scores of the model
        
        y_true = True labels
        y_pred = predicted labels
        
        """
        pass
    
#this is my strategy    
class MSE(Evaluation):
    """
    This uses mean square error
    
    y_true = np.ndarray
    y_test = np.ndarray
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        
        """
        This will calculate the mean squared error using sklearn
        """
        try:
            score = mean_squared_error(y_true,y_pred)
            logger.info("MSE {}".format(score))
            return score
        except Exception as e:
            logger.error("cannot calculate score {}".format(e))
            raise e
        

class R2(Evaluation):
    "This uses R2 score"
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        
        try:
            logger.info("Calculating R2 score")
            r2 = r2_score(y_true, y_pred)
            logger.info("r2 {}".format(r2))
            return r2
        except Exception as e:
            logger.error("Cannot calculate score {}".format(e))
            raise e
        

class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logger.info("Entered the calculate_score method of the RMSE class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logger.info("The root mean squared error value is: " + str(rmse))
            return rmse
        except Exception as e:
            logger.error(
                "Exception occurred in calculate_score method of the RMSE class. Exception message:  "
                + str(e)
            )
            raise e