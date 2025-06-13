import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import current_timestamp
import logging
from utils import download_online_retail_data

def initialize_spark_session_for_sub_module(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

def main():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        from utils import setup_logging
        logger = setup_logging(os.path.join("/app/logs", "data_ingestion.log"))

    spark = initialize_spark_session_for_sub_module("RetailDataIngestion")
    logger.info("SparkSession initialized for Data Ingestion.")

    raw_data_dir = "/app/data/raw/"
    bronze_path = "/app/data/bronze/online_retail_transactions"
    os.makedirs(raw_data_dir, exist_ok=True)
    raw_csv_path = download_online_retail_data(output_dir=raw_data_dir)
    logger.info(f"Raw data prepared at: {raw_csv_path}")
    retail_schema = StructType([
        StructField("InvoiceNo", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", StringType(), True),
        StructField("UnitPrice", FloatType(), True),
        StructField("CustomerID", FloatType(), True),
        StructField("Country", StringType(), True)
    ])
    logger.info(f"Reading CSV from {raw_csv_path} into Spark DataFrame...")
    raw_spark_df = spark.read.option("header", "true").option("inferSchema", "false").schema(retail_schema).csv(raw_csv_path)

    logger.info("Raw Spark DataFrame created. Schema:")
    raw_spark_df.printSchema()
    logger.info(f"Raw data row count: {raw_spark_df.count()}")
    raw_spark_df = raw_spark_df.withColumn("ingestion_timestamp", current_timestamp())
    logger.info(f"Writing data to Bronze layer at {bronze_path}...")
    raw_spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(bronze_path)
    logger.info("Data successfully written to Bronze layer.")