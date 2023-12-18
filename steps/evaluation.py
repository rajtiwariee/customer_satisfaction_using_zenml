from logs.customlog import logger
import pandas as pd
import mlflow
from typing_extensions import Annotated
from zenml import step
from src.evaluation import R2, MSE, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model : RegressorMixin, X_test : pd.DataFrame, y_test : pd.DataFrame) -> Tuple[
                                                
                                                Annotated[float, "MSE"],
                                                Annotated[float, "R2"],
                                                Annotated[float, "RMSE"]
                                            ]:
    
    """
    Evaluates the model on the test data and make prediction and calculate the score
    """
    try:
        y_pred = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("r2", r2)
        rmse_class = RMSE() 
        rmse = rmse_class.calculate_scores(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        return mse,r2,rmse
    except Exception as e:
        logger.error('Cannot evaluate the model {}'.format(e))
        
        raise e
    
    
    