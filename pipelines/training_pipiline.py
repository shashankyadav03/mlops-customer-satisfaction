from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model

@pipeline
def training_pipeline(path: str) -> None:
    """
    Training pipeline.
    
    Args:


    """
    df = ingest_data(path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)