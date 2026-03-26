from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import os
import json
import subprocess


# =========================
# CONFIG
# =========================
PYTHON_BIN = os.getenv("PYTHON_BIN", "python")
SCRIPTS_DIR = os.getenv("SCRIPTS_DIR", "/opt/airflow/scripts")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:8080")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME_BRAND", "Vehicle_Brand_Training")
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "s3://alpr-data")

DEFAULT_BRAND_EPOCHS = int(os.getenv("BRAND_EPOCHS", "15"))
DEFAULT_BRAND_IMGSZ = int(os.getenv("BRAND_IMGSZ", "224"))


# =========================
# TASKS
# =========================
def get_brand_retraining_candidates(**context):
    """
    Plus tard :
    - récupérer les images annotées avec la marque
    - filtrer les labels valides
    """
    candidates = {
        "source": "vehicle_brand_labeled_images",
        "num_samples": 1200,
        "num_classes": 18,
    }
    context["ti"].xcom_push(key="candidates", value=candidates)
    print(f"[get_brand_retraining_candidates] {candidates}")


def build_brand_dataset_version(**context):
    candidates = context["ti"].xcom_pull(task_ids="get_brand_retraining_candidates", key="candidates")

    dataset_version = f"brand_dataset_{datetime.utcnow().strftime('%Y_%m_%d_%H%M%S')}"
    dataset_uri = f"{S3_DATA_BUCKET}/brand_datasets/{dataset_version}/"

    manifest = {
        "dataset_version": dataset_version,
        "dataset_uri": dataset_uri,
        "num_samples": candidates["num_samples"],
        "num_classes": candidates["num_classes"],
        "created_at_utc": datetime.utcnow().isoformat(),
        "task": "vehicle_brand_classification",
    }

    context["ti"].xcom_push(key="dataset_manifest", value=manifest)
    print(f"[build_brand_dataset_version] {manifest}")


def train_vehicle_brand_model(**context):
    """
    Appelle le script train_vehicle_brand_model.py
    qui doit :
    - entraîner le classifieur marque
    - logguer dans MLflow
    - écrire metrics.json et run_info.json
    """
    manifest = context["ti"].xcom_pull(
        task_ids="build_brand_dataset_version",
        key="dataset_manifest"
    )

    if not manifest:
        raise ValueError("dataset_manifest introuvable dans XCom")

    output_dir = f"/tmp/{manifest['dataset_version']}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        PYTHON_BIN,
        f"{SCRIPTS_DIR}/train_vehicle_brand_model.py",
        "--dataset-uri", manifest["dataset_uri"],
        "--dataset-version", manifest["dataset_version"],
        "--output-dir", output_dir,
        "--epochs", str(DEFAULT_BRAND_EPOCHS),
        "--imgsz", str(DEFAULT_BRAND_IMGSZ),
        "--mlflow-tracking-uri", MLFLOW_TRACKING_URI,
        "--mlflow-experiment", MLFLOW_EXPERIMENT_NAME,
    ]

    print("[train_vehicle_brand_model] Running command:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True
    )

    print("[train_vehicle_brand_model] Return code:", result.returncode)
    print("[train_vehicle_brand_model] STDOUT:")
    print(result.stdout)
    print("[train_vehicle_brand_model] STDERR:")
    print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            "train_vehicle_brand_model.py a échoué.\n"
            f"Return code: {result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    metrics_path = os.path.join(output_dir, "metrics.json")
    run_info_path = os.path.join(output_dir, "run_info.json")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    run_info = {}
    if os.path.exists(run_info_path):
        with open(run_info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)

    context["ti"].xcom_push(key="metrics", value=metrics)
    context["ti"].xcom_push(key="run_info", value=run_info)

    print(f"[train_vehicle_brand_model] metrics={metrics}")
    print(f"[train_vehicle_brand_model] run_info={run_info}")


def mark_brand_candidate_ready(**context):
    run_info = context["ti"].xcom_pull(task_ids="train_vehicle_brand_model", key="run_info")
    print(f"[mark_brand_candidate_ready] run_info={run_info}")


def notify_brand_summary(**context):
    manifest = context["ti"].xcom_pull(task_ids="build_brand_dataset_version", key="dataset_manifest")
    metrics = context["ti"].xcom_pull(task_ids="train_vehicle_brand_model", key="metrics")
    run_info = context["ti"].xcom_pull(task_ids="train_vehicle_brand_model", key="run_info")

    print("===== BRAND TRAINING SUMMARY =====")
    print(f"Dataset version: {manifest['dataset_version']}")
    print(f"Dataset uri: {manifest['dataset_uri']}")
    print(f"Run info: {run_info}")
    print(f"Metrics: {metrics}")


# =========================
# DAG
# =========================
with DAG(
    dag_id="vehicle_brand_training_pipeline",
    start_date=datetime(2026, 3, 1),
    schedule=None,  # manuel au début
    catchup=False,
    tags=["alpr", "vehicle_brand", "training", "mlflow"],
) as dag:

    start = EmptyOperator(task_id="start")

    t1 = PythonOperator(
        task_id="get_brand_retraining_candidates",
        python_callable=get_brand_retraining_candidates,
    )

    t2 = PythonOperator(
        task_id="build_brand_dataset_version",
        python_callable=build_brand_dataset_version,
    )

    t3 = PythonOperator(
        task_id="train_vehicle_brand_model",
        python_callable=train_vehicle_brand_model,
    )

    t4 = PythonOperator(
        task_id="mark_brand_candidate_ready",
        python_callable=mark_brand_candidate_ready,
    )

    t5 = PythonOperator(
        task_id="notify_brand_summary",
        python_callable=notify_brand_summary,
    )

    start >> t1 >> t2 >> t3 >> t4 >> t5