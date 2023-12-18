from pipelines.training_pipeline import train_pipeline
from zenml.client import Client



if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path = 'X:\Projects_ML_DL_LLM\Customer_Satisfaction\data\olist_customers_dataset.csv')
    # mlflow ui --backend-store-uri "file:C:\Users\Raj\AppData\Roaming\zenml\local_stores\179ca842-ba86-415c-bc17-caa46c4c0c05\mlruns"