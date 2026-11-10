"""
DAG : réentraînement automatique du classificateur de marques.

    check_trigger --> retrain_cycle --> report

Commandes exécutées DANS le conteneur alpr-engine via l'API Docker (alpr_common.run_in_engine)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator, ShortCircuitOperator

from alpr_common import run_in_engine   # commun aux DAG ALPR (airflow/dags/alpr_common.py)


def check_trigger(**context) -> bool:
    """Assez de nouvelles images validées (ou un challenger à juger) ? Sinon, le DAG s'arrête ici."""
    decision = run_in_engine("src.mlops.continuous", "--check")
    context["ti"].xcom_push(key="decision", value=decision)
    print(f"Décision : {decision['reason']}")
    return bool(decision["run"])


def retrain_cycle(**context) -> None:
    """Cycle complet : entraînement -> challenger -> promotion (garde-fous inchangés)."""
    report = run_in_engine("src.mlops.continuous")
    context["ti"].xcom_push(key="report", value=report)
    if report.get("status") in ("error", "locked"):
        raise AirflowException(f"Cycle {report['status']} : {report.get('reason')}")


def report(**context) -> None:
    """Résumé lisible de l'exécution (et point d'accroche d'une future notification)."""
    result = context["ti"].xcom_pull(task_ids="retrain_cycle", key="report") or {}
    promotion = result.get("promotion") or {}
    print(f"Statut : {result.get('status')}")
    print(f"Déclenchement : {(result.get('trigger') or {}).get('reason')}")
    if result.get("training"):
        print(f"Challenger : v{result['training'].get('version')} — métriques {result['training'].get('metrics')}")
    if promotion:
        print(f"Promotion : {'oui' if promotion.get('promoted') else 'non'} — {promotion.get('reason')}")


with DAG(
    dag_id="alpr_brand_retraining",
    description="Réentraînement automatique du classificateur de marque (validations → challenger → promotion)",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,                   # on ne rattrap pas les exec des jours passés
    max_active_runs=1,               # un seul cycle à la fois (en plus du verrou du cycle)
    dagrun_timeout=timedelta(hours=2),
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["alpr", "mlops", "retraining"],
) as dag:
    check = ShortCircuitOperator(task_id="check_trigger", python_callable=check_trigger)
    cycle = PythonOperator(task_id="retrain_cycle", python_callable=retrain_cycle)
    summary = PythonOperator(task_id="report", python_callable=report)

    check >> cycle >> summary
