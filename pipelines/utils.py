import logging
import pandas as pd

from src.data_cleaning import DataCleaning, DataPreProcessStrategy

def get_data_for_test() -> pd.DataFrame:
    """Get data for test."""
    try:
        df = pd.read_csv(
            "/Users/admin/Downloads/mlops-customer-satisfaction/data/olist_customers_dataset.csv"
        )
        df = df.sample(n=100)
        preprocessing_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocessing_strategy)
        df = data_cleaning.execute_strategy(df)
        df.drop(["review_score"], inplace=True, axis=1)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(f"Error: {e}")
        return None
