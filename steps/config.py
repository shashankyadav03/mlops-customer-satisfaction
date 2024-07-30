from zenml.steps import BaseParameters

class ModelNameCofig(BaseParameters):
    """
    Model name configuration.
    """
    model_name: str = "LinearRegression"