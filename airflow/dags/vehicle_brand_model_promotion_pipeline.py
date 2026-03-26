from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import os
import mlflow
from mlflow.tracking import MlflowClient


# =========================
# CONFIG
# =========================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-service:8080")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME_BRAND", "Vehicle_Brand_Training")
MODEL_NAME = os.getenv("VEHICLE_BRAND_MODEL_NAME", "vehicle_brand_model")


# =========================
# HELPERS
# =========================
def get_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_experiment_id(client):
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise ValueError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")
    return exp.experiment_id


def safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


# =========================
# TASKS
# =========================
def load_latest_ready_brand_candidate(**context):
    """
    Cherche le dernier run marque prêt à promouvoir.
    Le script train_vehicle_brand_model.py doit avoir posé :
    - tags.candidate = true
    - tags.candidate_status = READY_FOR_PROMOTION
    - tags.model_name = vehicle_brand_model
    """
    client = get_client()
    experiment_id = get_experiment_id(client)

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.candidate = 'true' "
            f"AND tags.candidate_status = 'READY_FOR_PROMOTION' "
            f"AND tags.model_name = '{MODEL_NAME}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No READY_FOR_PROMOTION brand candidate found.")

    run = runs[0]
    candidate = {
        "run_id": run.info.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }

    context["ti"].xcom_push(key="candidate", value=candidate)
    print(f"[load_latest_ready_brand_candidate] run_id={candidate['run_id']}")


def load_current_brand_production_model(**context):
    """
    Charge la version Production du modèle marque si elle existe.
    Sinon baseline minimale.
    """
    client = get_client()

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        versions = []

    prod_version = None
    for mv in versions:
        if mv.current_stage == "Production":
            prod_version = mv
            break

    if prod_version is None:
        production_reference = {
            "has_production_model": False,
            "version": None,
            "run_id": None,
            "metrics": {
                "accuracy": 0.0,
                "top_3_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            },
        }
    else:
        run = client.get_run(prod_version.run_id)
        production_reference = {
            "has_production_model": True,
            "version": prod_version.version,
            "run_id": prod_version.run_id,
            "metrics": run.data.metrics,
        }

    context["ti"].xcom_push(key="production_reference", value=production_reference)
    print(f"[load_current_brand_production_model] {production_reference}")


def compare_brand_candidate_vs_production(**context):
    candidate = context["ti"].xcom_pull(task_ids="load_latest_ready_brand_candidate", key="candidate")
    production_reference = context["ti"].xcom_pull(
        task_ids="load_current_brand_production_model",
        key="production_reference",
    )

    c_metrics = candidate["metrics"]
    p_metrics = production_reference["metrics"]

    c_accuracy = safe_float(c_metrics.get("accuracy"), 0.0)
    c_top3 = safe_float(c_metrics.get("top_3_accuracy"), 0.0)
    c_precision = safe_float(c_metrics.get("precision"), 0.0)
    c_recall = safe_float(c_metrics.get("recall"), 0.0)
    c_f1 = safe_float(c_metrics.get("f1_score"), 0.0)

    p_accuracy = safe_float(p_metrics.get("accuracy"), 0.0)
    p_top3 = safe_float(p_metrics.get("top_3_accuracy"), 0.0)
    p_precision = safe_float(p_metrics.get("precision"), 0.0)
    p_recall = safe_float(p_metrics.get("recall"), 0.0)
    p_f1 = safe_float(p_metrics.get("f1_score"), 0.0)

    promote = (
        c_accuracy >= p_accuracy
        and c_top3 >= p_top3
        and c_f1 >= p_f1
    )

    comparison = {
        "promote": promote,
        "candidate_metrics": {
            "accuracy": c_accuracy,
            "top_3_accuracy": c_top3,
            "precision": c_precision,
            "recall": c_recall,
            "f1_score": c_f1,
        },
        "production_metrics": {
            "accuracy": p_accuracy,
            "top_3_accuracy": p_top3,
            "precision": p_precision,
            "recall": p_recall,
            "f1_score": p_f1,
        },
    }

    context["ti"].xcom_push(key="comparison", value=comparison)
    print(f"[compare_brand_candidate_vs_production] {comparison}")


def brand_promotion_decision(**context):
    comparison = context["ti"].xcom_pull(task_ids="compare_brand_candidate_vs_production", key="comparison")
    return "register_and_promote_brand_model" if comparison["promote"] else "reject_brand_candidate"


def register_and_promote_brand_model(**context):
    client = get_client()
    candidate = context["ti"].xcom_pull(task_ids="load_latest_ready_brand_candidate", key="candidate")
    run_id = candidate["run_id"]

    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    new_version = registered_model.version

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        versions = []

    for mv in versions:
        if mv.current_stage == "Production":
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=mv.version,
                stage="Archived",
            )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version,
        stage="Production",
    )

    client.set_tag(run_id, "candidate_status", "PROMOTED")
    client.set_tag(run_id, "promoted_model_version", str(new_version))

    context["ti"].xcom_push(key="promoted_version", value=str(new_version))
    print(f"[register_and_promote_brand_model] run_id={run_id}, version={new_version}")


def reject_brand_candidate(**context):
    client = get_client()
    candidate = context["ti"].xcom_pull(task_ids="load_latest_ready_brand_candidate", key="candidate")
    run_id = candidate["run_id"]

    client.set_tag(run_id, "candidate_status", "REJECTED")
    print(f"[reject_brand_candidate] run_id={run_id} rejected")


def write_brand_decision_log(**context):
    candidate = context["ti"].xcom_pull(task_ids="load_latest_ready_brand_candidate", key="candidate")
    comparison = context["ti"].xcom_pull(task_ids="compare_brand_candidate_vs_production", key="comparison")
    promoted_version = context["ti"].xcom_pull(task_ids="register_and_promote_brand_model", key="promoted_version")

    print("===== BRAND PROMOTION SUMMARY =====")
    print(f"Candidate run_id: {candidate['run_id']}")
    print(f"Promote: {comparison['promote']}")
    print(f"Candidate metrics: {comparison['candidate_metrics']}")
    print(f"Production metrics: {comparison['production_metrics']}")
    print(f"Promoted version: {promoted_version}")


# =========================
# DAG
# =========================
with DAG(
    dag_id="vehicle_brand_model_promotion_pipeline",
    start_date=datetime(2026, 3, 1),
    schedule=None,
    catchup=False,
    tags=["alpr", "vehicle_brand", "promotion", "mlflow"],
) as dag:

    start = EmptyOperator(task_id="start")

    t1 = PythonOperator(
        task_id="load_latest_ready_brand_candidate",
        python_callable=load_latest_ready_brand_candidate,
    )

    t2 = PythonOperator(
        task_id="load_current_brand_production_model",
        python_callable=load_current_brand_production_model,
    )

    t3 = PythonOperator(
        task_id="compare_brand_candidate_vs_production",
        python_callable=compare_brand_candidate_vs_production,
    )

    t4 = BranchPythonOperator(
        task_id="brand_promotion_decision",
        python_callable=brand_promotion_decision,
    )

    t5 = PythonOperator(
        task_id="register_and_promote_brand_model",
        python_callable=register_and_promote_brand_model,
    )

    t6 = PythonOperator(
        task_id="reject_brand_candidate",
        python_callable=reject_brand_candidate,
    )

    t7 = PythonOperator(
        task_id="write_brand_decision_log",
        python_callable=write_brand_decision_log,
        trigger_rule="none_failed_min_one_success",
    )

    start >> t1 >> t2 >> t3 >> t4
    t4 >> t5 >> t7
    t4 >> t6 >> t7