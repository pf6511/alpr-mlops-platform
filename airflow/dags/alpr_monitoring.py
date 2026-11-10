"""
DAG : monitoring quotidien de la solution en production.

    monitoring_report --> check_alerts

monitoring_report : python -m src.monitoring.report dans alpr-engine (indicateurs,
                    seuils du cahier des charges, rapport de dérive Evidently)
check_alerts      : en échec (rouge) si un seuil est dépassé → visible dans Airflow,
                    point d'accroche d'une notification (e-mail, Slack...)

DAG séparé du réentraînement : le monitoring doit tourner chaque jour, même quand
il n'y a rien à réentraîner (le DAG de réentraînement s'arrête alors à check_trigger).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator

from alpr_common import run_in_engine   # commun aux DAG ALPR (airflow/dags/alpr_common.py)


def monitoring_report(**context) -> None:
    """Calcule le rapport de monitoring et le transmet à la tâche suivante (XCom)."""
    report = run_in_engine("src.monitoring.report")
    context["ti"].xcom_push(key="report", value=report)
    recent = report.get("recente", {})
    print(f"Passages (période récente) : {recent.get('passages')}")
    print(f"Taux d'erreur du modèle de marque : {recent.get('taux_erreur_marque')}")
    print(f"Dérive : {report.get('derive')}")
    print(f"Rapport HTML : {report.get('rapport_html')}")


def check_alerts(**context) -> None:
    """Échoue si un critère d'acceptation n'est plus respecté (alerte visible dans Airflow)."""
    report = context["ti"].xcom_pull(task_ids="monitoring_report", key="report") or {}
    alerts = report.get("alertes", [])
    if alerts:
        raise AirflowException("Seuils dépassés : " + " ; ".join(alerts))
    print("Tous les critères d'acceptation sont respectés.")


with DAG(
    dag_id="alpr_monitoring",
    description="Monitoring quotidien : indicateurs, seuils du cahier des charges, dérive (Evidently)",
    schedule="@daily",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=30),
    tags=["alpr", "mlops", "monitoring"],
) as dag:
    report_task = PythonOperator(task_id="monitoring_report", python_callable=monitoring_report,
                                 retries=1, retry_delay=timedelta(minutes=5))
    alerts_task = PythonOperator(task_id="check_alerts", python_callable=check_alerts,
                                 retries=0)       # une alerte ne disparaît pas en réessayant

    report_task >> alerts_task
