from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    
    "Model Configs"
    
    model_name : str = "LinearRegression"
    
class Datapath(BaseParameters):
    
    "data path"
    
    data_path : str = 'data/olist_customers_dataset.csv'    