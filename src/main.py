import os
import sys
import logging
import warnings
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_ingestion import main as run_data_ingestion
from etl import main as run_etl
from model_training import main as run_model_training
from utils import setup_logging
import mlflow
warnings.filterwarnings(
    "ignore", 
    message="When packaging an MLflow Model that depends on MLflow, .*"
)

def main():
    log_file_path = os.path.join("/app/logs", "pipeline.log")
    logger = setup_logging(log_file_path)
    logger.info("--- Starting the entire Spark Lakehouse Pipeline ---")
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow Tracking URI set to: {mlflow_tracking_uri}")

    logger.info("\n--- Step 1: Running Data Ingestion ---")
    run_data_ingestion() 
    logger.info("--- Data Ingestion Completed ---")
    logger.info("\n--- Step 2: Running ETL Pipeline ---")
    run_etl()
    logger.info("--- ETL Pipeline Completed ---")
    logger.info("\n--- Step 3: Running Model Training ---")
    run_model_training()
    logger.info("--- Model Training Completed ---")

    logger.info("\n--- All Spark Lakehouse Pipeline steps completed successfully! ---")

if __name__ == "__main__":
    main()