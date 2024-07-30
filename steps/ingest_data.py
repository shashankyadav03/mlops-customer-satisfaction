import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingest data from a given path.
    """
    def __init__(self, path: str):
        """
        Args:
            path: Path to the data.
        """
        self.path = path

    def get_data(self):
        """
        Read data from the path.
        """
        logging.info(f"Reading data from {self.path}")
        return pd.read_csv(self.path)
    
@step
def ingest_data(path: str) -> pd.DataFrame:
    """
    Ingest data from a data path.

    Args:
        data_path: Path to the data.

    Returns:
        Dataframe containing the data.
    """ 
    try:
        ingest_data = IngestData(path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in Ingesting data: {e}")
        raise e
    
