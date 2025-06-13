    FROM apache/spark-py:latest

    USER root

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY src/ /app/src/

    ENV SPARK_HOME=/opt/spark
    ENV PATH=$SPARK_HOME/bin:$PATH
    ENV PYTHONPATH=/app:$PYTHONPATH

    RUN mkdir -p /app/data/bronze /app/data/silver /app/data/gold /app/logs /app/mlruns
    RUN chmod -R 777 /app/mlruns
    CMD ["spark-submit", "--packages", "io.delta:delta-core_2.12:2.4.0", "--conf", "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension", "--conf", "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog", "--conf", "spark.databricks.delta.optimize.zorder.checkStatsCollection.enabled=false", "--master", "local[*]", "--driver-memory", "4g", "--executor-memory", "4g", "/app/src/main.py"]# Correct: Spark connects to MLflow's internal port