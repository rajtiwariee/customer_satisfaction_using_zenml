from zenml.pipelines import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from logs.customlog import logger

@pipeline  #(enable_cache = True) zenml supports cache version if there are no changes in the code it will use the previous version
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test , y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test , y_train, y_test)
    logger.info(f'>>Training pipeline executed successfully..')
    mse, r2, rmse = evaluate_model(model,X_test, y_test)