import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, count, trim, countDistinct
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, LongType, IntegerType
from delta.tables import DeltaTable
import logging

def initialize_spark_session_for_sub_module(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

def main():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        from utils import setup_logging
        logger = setup_logging(os.path.join("/app/logs", "etl.log"))

    spark = initialize_spark_session_for_sub_module("RetailLakehouseETL")
    logger.info("SparkSession initialized for ETL.")

    bronze_path = "/app/data/bronze/online_retail_transactions"
    silver_path = "/app/data/silver/cleaned_online_retail_transactions"
    gold_path = "/app/data/gold/aggregated_online_retail_data"

    logger.info(f"Reading data from Bronze layer: {bronze_path}")
    bronze_df = spark.read.format("delta").load(bronze_path)
    logger.info("Bronze data loaded successfully. Schema:")
    bronze_df.printSchema()
    logger.info(f"Bronze layer row count: {bronze_df.count()}") 
    if len(bronze_df.columns) == 0:
        logger.error("Bronze DataFrame has no columns after loading. This indicates an issue with the source data or initial ingestion.")
        raise ValueError("Bronze DataFrame has an empty schema.")
    if bronze_df.count() == 0:
        logger.warning("Bronze DataFrame is empty after loading. This will result in empty Silver and Gold layers.")

    logger.info("--- Starting Data Cleaning and Transformation for Silver Layer ---")

    cleaned_df = bronze_df
    if "InvoiceDate" in cleaned_df.columns: 
        cleaned_df = cleaned_df.drop("InvoiceDate")
        logger.info("InvoiceDate column dropped.")
    else:
        logger.warning("InvoiceDate column not found in DataFrame. Skipping drop.")
    logger.info("Schema after dropping InvoiceDate:")
    cleaned_df.printSchema()
    logger.info(f"Columns remaining: {cleaned_df.columns}")
    if "Quantity" in cleaned_df.columns:
        cleaned_df = cleaned_df.withColumn("Quantity", col("Quantity").cast(IntegerType()))
        logger.info("Quantity cast to IntegerType.")
    else:
        logger.warning("Quantity column not found in DataFrame.")
        
    if "UnitPrice" in cleaned_df.columns:
        cleaned_df = cleaned_df.withColumn("UnitPrice", col("UnitPrice").cast(DoubleType()))
        logger.info("UnitPrice cast to DoubleType.")
    else:
        logger.warning("UnitPrice column not found in DataFrame.")
    cleaned_df = cleaned_df.filter(col("Quantity").isNotNull()).filter(col("Quantity") > 0)
    logger.info(f"Rows after Quantity > 0 filter and not null: {cleaned_df.count()}")
    cleaned_df = cleaned_df.filter(col("UnitPrice").isNotNull()).filter(col("UnitPrice") > 0)
    logger.info(f"Rows after UnitPrice > 0 filter and not null: {cleaned_df.count()}")
    cleaned_df = cleaned_df.na.drop(subset=["CustomerID"])
    logger.info(f"Rows after dropping null CustomerID: {cleaned_df.count()}")
    if "CustomerID" in cleaned_df.columns:
        cleaned_df = cleaned_df.withColumn("CustomerID", col("CustomerID").cast(LongType()))
        logger.info("CustomerID cast to LongType.")
    else:
        logger.warning("CustomerID column not found in DataFrame.")
    for c in ["InvoiceNo", "StockCode", "Description", "Country"]:
        if c in cleaned_df.columns and cleaned_df.schema[c].dataType.simpleString() == "string":
            cleaned_df = cleaned_df.withColumn(c, trim(col(c)))
    logger.info("Whitespace trimmed from string columns.")
    logger.info("Schema before final column selection for Silver layer:")
    cleaned_df.printSchema()
    logger.info(f"Columns present before final selection: {cleaned_df.columns}")
    desired_silver_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "UnitPrice",
        "CustomerID",
        "Country",
        "ingestion_timestamp"
    ]
    final_select_columns = [c for c in desired_silver_columns if c in cleaned_df.columns]
    if not final_select_columns:
        logger.error(f"No desired columns found in the DataFrame to write to Silver layer. Desired: {desired_silver_columns}, Actual: {cleaned_df.columns}")
        raise ValueError("DataFrame has an empty schema for Silver layer after column selection. Cannot write.")

    cleaned_df = cleaned_df.select([col(c) for c in final_select_columns]) 
    logger.info("Silver layer schema after transformations and final selection:")
    cleaned_df.printSchema()
    logger.info(f"Final rows after all cleaning steps for Silver layer: {cleaned_df.count()}")

    logger.info(f"Writing data to Silver layer at {silver_path}...")
    if "Country" in cleaned_df.columns:
        cleaned_df.repartition(F.lit(5), col("Country")).write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    else:
        logger.warning("Country column not found for partitioning. Writing without partitioning by Country.")
        cleaned_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(silver_path)
    logger.info("Data successfully written to Silver layer.")
    logger.info("Silver layer row count: %d", cleaned_df.count())
    logger.info("Performing aggregations for Gold layer...")
    gold_aggregation_columns = ["Country", "InvoiceNo", "Quantity", "UnitPrice", "CustomerID", "StockCode"]
    missing_gold_columns = [c for c in gold_aggregation_columns if c not in cleaned_df.columns]
    if missing_gold_columns:
        logger.error(f"Missing columns required for Gold layer aggregation: {missing_gold_columns}. Cannot perform aggregation.")
        raise ValueError("Missing columns for Gold layer aggregation.")

    gold_df = cleaned_df.groupBy("Country").agg(count("InvoiceNo").alias("TotalTransactions"),
                                                sum(col("Quantity") * col("UnitPrice")).alias("TotalRevenue"),
                                                sum("Quantity").alias("TotalQuantitySold"),
                                                count(col("StockCode")).alias("TotalItemsSold"), 
                                                countDistinct(col("CustomerID")).alias("UniqueCustomers"),
                                                countDistinct(col("StockCode")).alias("UniqueProductsSold")
                                            ).orderBy("Country")

    logger.info("Aggregations complete. Gold layer schema:")
    gold_df.printSchema()
    logger.info("Gold layer row count: %d", gold_df.count())
    logger.info(f"Writing data to Gold layer at {gold_path} with optimization...")
    if "Country" in gold_df.columns:
        gold_df.write.format("delta").mode("overwrite").partitionBy("Country").option("delta.dataSkippingNumIndexedCols", 2).save(gold_path)
    else:
        logger.warning("Country column not found in Gold DataFrame. Writing without partitioning by Country.")
        gold_df.write.format("delta").mode("overwrite").option("delta.dataSkippingNumIndexedCols", 2).save(gold_path)

    logger.info("Gold layer written.")
    delta_gold_table = DeltaTable.forPath(spark, gold_path)
    logger.info("Optimizing Gold Delta table with ZORDER by TotalRevenue...")
    if "TotalRevenue" in gold_df.columns:
        delta_gold_table.optimize().executeZOrderBy("TotalRevenue")
        logger.info("Gold Delta table optimized.")
    else:
        logger.warning("TotalRevenue column not found in Gold Delta table. Skipping ZORDER optimization.")