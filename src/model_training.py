import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType, DoubleType
import mlflow
import logging

def initialize_spark_session_for_sub_module(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

def main():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        from utils import setup_logging
        logger = setup_logging(os.path.join("/app/logs", "model_training.log"))

    spark = initialize_spark_session_for_sub_module("RetailModelTraining")
    logger.info("SparkSession initialized for model training.")
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow_server:5000"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    logger.info(f"MLflow Tracking URI set to: {mlflow.get_tracking_uri()}")
    mlflow.set_experiment("Online_Retail_Quantity_Prediction")
    logger.info(f"MLflow experiment configured as: Online_Retail_Quantity_Prediction")

    silver_path = "/app/data/silver/cleaned_online_retail_transactions"

    logger.info(f"Reading data from Silver layer for ML: {silver_path}")
    data = spark.read.format("delta").load(silver_path)
    logger.info("Silver data loaded successfully for ML. Schema:")
    data.printSchema()
    logger.info(f"Silver layer row count for ML: {data.count()}")
    data = data.withColumn("Quantity", col("Quantity").cast(IntegerType()))
    data = data.withColumn("UnitPrice", col("UnitPrice").cast(DoubleType()))
    logger.info("Quantity and UnitPrice columns cast to numeric types.")
    data.printSchema()
    indexer_country = StringIndexer(inputCol="Country", outputCol="CountryIndex")
    data = indexer_country.fit(data).transform(data)
    logger.info("Country indexed to CountryIndex.")
    indexer_stockcode = StringIndexer(inputCol="StockCode", outputCol="StockCodeIndex")
    data = indexer_stockcode.fit(data).transform(data)
    logger.info("StockCode indexed to StockCodeIndex.")
    feature_columns = [
        "CustomerID",
        "UnitPrice",
        "CountryIndex",
        "StockCodeIndex"
    ]
    logger.info(f"Feature columns for the model: {feature_columns}")

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    assembled_data = assembler.transform(data)
    logger.info("Features assembled in 'features' column.")

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=False)
    if assembled_data.isEmpty():
        logger.warning("Assembled data is empty. Skipping model training.")
        mlflow.end_run(status="FAILED")
        return
    
    scaler_model = scaler.fit(assembled_data)
    scaled_data = scaler_model.transform(assembled_data)
    logger.info("Features scaled in 'scaledFeatures' column.")
    (training_data, test_data) = scaled_data.randomSplit([0.8, 0.2], seed=42)
    logger.info(f"Data split: Training={training_data.count()} rows, Test={test_data.count()} rows.")


    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        n_estimators = 100
        max_depth = 15
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", feature_columns)
        logger.info(f"Logged parameters: n_estimators={n_estimators}, max_depth={max_depth}")
        logger.info("Training RandomForestRegressor model...")
        rf = RandomForestRegressor(featuresCol="scaledFeatures", labelCol="Quantity",
                                    numTrees=n_estimators, maxDepth=max_depth, seed=42)
        rf_model = rf.fit(training_data)
        logger.info("Model training completed.")
        predictions = rf_model.transform(test_data)
        logger.info("Predictions generated.")
        evaluator_rmse = RegressionEvaluator(labelCol="Quantity", predictionCol="prediction", metricName="rmse")
        evaluator_r2 = RegressionEvaluator(labelCol="Quantity", predictionCol="prediction", metricName="r2")

        rmse = evaluator_rmse.evaluate(predictions)
        r2 = evaluator_r2.evaluate(predictions)

        logger.info(f"Root Mean Squared Error (RMSE) on test data = {rmse}")
        logger.info(f"R-squared (R2) on test data = {r2}")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        logger.info("Metrics logged: RMSE and R2.")
        logger.info("Logging model with MLflow...")
        mlflow.spark.log_model(
            spark_model=rf_model,
            artifact_path="random_forest_quantity_model",
            registered_model_name="OnlineRetailQuantityPredictor"
        )
        logger.info("Model logged successfully.")
        mlflow.end_run(status="FINISHED")