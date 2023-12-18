from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    
    "Model Configs"
    
    model_name : str = "LinearRegression"
    
class Datapath(BaseParameters):
    
    "data path"
    
    data_path : str = 'X:\Projects_ML_DL_LLM\Customer_Satisfaction\data\olist_customers_dataset.csv'    