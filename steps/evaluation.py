import logging
import pandas as pd
from zenml import step

@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluate the model on the given data.
    
    Args:
        df: Ingeted data.
    """
    logging.info("Evaluating model")
    # Your evaluation code here
    pass