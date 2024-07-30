from abc import ABC, abstractmethod
import logging
from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class for data cleaning strategies.
    """

    @abstractmethod
    def handle_data(self, path: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Concrete class for data cleaning strategies.
    """

    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.

        Args:
            df: Ingeted data.

        Returns:
            Preprocessed data.
        """
        logging.info("Preprocessing data")
        try:
            data =df.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["product_category_name"].fillna("Others", inplace=True)
            data["review_comment_message"].fillna("No Review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data.drop(cols_to_drop, axis=1, inplace=True)
            return data
        except Exception as e:
            logging.error(f"Error in Preprocessing data: {e}")
            raise e
        
class DataSplitStrategy(DataStrategy):
    """"
    Strategy for splitting data into train and test sets.
    """
    def handle_data(self, path: pd.DataFrame) -> Union[pd.DataFrame | pd.Series]:
        """
        Divide the data into training and testing sets.
        """
        try:
            X = path.drop("review_score", axis=1)
            y = path["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error in Splitting data: {e}")
            raise e
        
class DataCleaning:
    """
    Context class for data cleaning strategies.
    """
    def __init__(self,path: pd.DataFrame, strategy: DataStrategy):
        """
        Args:
            strategy: Data cleaning strategy.
        """
        self.data = path
        self.strategy = strategy

    def execute_strategy(self, path: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """"
        Execute the data cleaning strategy.
        """
        try:
            return self.strategy.handle_data(path)
        except Exception as e:
            logging.error(f"Error in executing strategy: {e}")
            raise e
        
# if __name__ == "__main__":
#     data = pd.read_csv("/Users/admin/Downloads/mlops-customer-satisfaction/data/olist_customers_dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data = data_cleaning.execute_strategy(data)