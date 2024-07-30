from pipelines.training_pipiline import training_pipeline
import logging
from zenml.client import Client

# Run the pipeline
def main():
    print("Client uri: ", Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(path='/Users/admin/Downloads/mlops-customer-satisfaction/data/olist_customers_dataset.csv')

if __name__ == '__main__':
    main()

# mlflow ui --backend-store-uri "file:/Users/admin/Library/Application Support/zenml/local_stores/e4f4fe28-0ba5-4353-b481-6f23e0ef5d2f/mlruns"