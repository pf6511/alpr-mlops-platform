FROM apache/airflow:2.10.0

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir \
    mlflow \
    boto3 \
    psycopg2-binary