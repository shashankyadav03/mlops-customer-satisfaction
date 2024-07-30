import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from steps.config import ModelNameCofig
@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameCofig
    ) -> RegressorMixin:
    """
    Train a model on the given data.
    
    Args:
        df: Ingeted data.
    """
    logging.info("Training model")
    # Your training code here
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            logging.error(f"Model name {config.model_name} not found")
            raise ValueError(f"Model name {config.model_name} not found")
    except Exception as e:
        logging.error(f"Error in Training model: {e}")
        raise e
    