import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class Evaluation(ABC):
    """
    Abstract class for evaluation strategies.
    """
    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the evaluation scores.
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
        Returns:
            Dictionary containing the evaluation scores.
        """
        pass

class MSE(Evaluation):
    """
    Concrete class for evaluation strategies.
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the evaluation scores.
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
        Returns:
            Dictionary containing the evaluation scores.
        """
        logging.info("Calculating MSE")
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in Calculating MSE: {e}")
            raise e
        
class R2(Evaluation):
    """
    Concrete class for evaluation strategies.
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the evaluation scores.
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
        Returns:
            Dictionary containing the evaluation scores.
        """
        logging.info("Calculating R2")
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in Calculating R2: {e}")
            raise e
    
class RMSE(Evaluation):
    """
    Concrete class for evaluation strategies.
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the evaluation scores.
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
        Returns:
            Dictionary containing the evaluation scores.
        """
        logging.info("Calculating RMSE")
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in Calculating RMSE: {e}")
            raise e