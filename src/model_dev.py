import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for model strategies.
    """
    
    @abstractmethod
    def train(self, X_train,y_train) -> None:
        """
        Train the model.
        Args:
            X_train: Training data.
            y_train: Target data.
        Returns:
            None
        """
        pass
    
class LinearRegressionModel(Model):
    """
    Concrete class for model strategies.
    """
    
    def train(self, X_train, y_train,**kwargs) -> None:
        """
        Train the model.
        Args:
            X_train: Training data.
            y_train: Target data.
        Returns:
            None
        """
        logging.info("Training Linear Regression Model")
        try:
            # Your training code here
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Trained")
            return reg
        except Exception as e:
            logging.error(f"Error in Training Linear Regression Model: {e}")
            raise e