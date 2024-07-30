import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.DataFrame, 'y_train'],
    Annotated[pd.DataFrame, 'y_test']
    ]:
    """
    Clean the data.
    
    Args:
        df: Ingeted data.
    
    Returns:
        Tuple of cleaned data.
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.execute_strategy()

        divide_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.execute_strategy()
        logging.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    except Exception as e:
        logging.error(f"Error in Cleaning data: {e}")
        raise e