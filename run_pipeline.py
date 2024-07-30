from pipelines.training_pipiline import training_pipeline
import logging

# Run the pipeline
def main():
    training_pipeline(path='/Users/admin/Downloads/mlops-customer-satisfaction/data/olist_customers_dataset.csv')

if __name__ == '__main__':
    main()