"""
DAG : réentraînement automatique du classificateur de marque.

    check_trigger ──► retrain_cycle ──► report

Airflow décide QUAND (planification, reprises, historique des exécutions).
Le code du projet décide QUOI et SI : chaque tâche appelle la même commande
que le service alpr-retrain (python -m src.mlops.continuous), aucune règle
métier n'est réécrite ici.

Les commandes sont exécutées DANS le conteneur alpr-engine (qui contient le code,
les modèles et la configuration) via l'API Docker : l'image Airflow reste légère.
Compromis de développement : Airflow accède au socket Docker. En production :
KubernetesPodOperator (un pod avec l'image de l'application) ou une image
Airflow dédiée, avec des droits limités.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator, ShortCircuitOperator

ENGINE_CONTAINER = os.environ.get("ALPR_ENGINE_CONTAINER", "alpr-engine")
RESULT_PREFIX = "ALPR_RESULT "          # dernière ligne de sortie de src.mlops.continuous


def run_in_engine(*module_args: str) -> dict:
    """Exécute « python -m <module> <args> » dans le conteneur de l'application et renvoie son résultat."""
    import docker

    container = docker.from_env().containers.get(ENGINE_CONTAINER)
    exit_code, (stdout, stderr) = container.exec_run(
        ["python", "-m", *module_args], workdir="/app", demux=True
    )
    out = (stdout or b"").decode("utf-8", "replace")
    err = (stderr or b"").decode("utf-8", "replace")
    print(out)
    if err:
        print(err)                           # journaux de la commande (logging)

    results = [line[len(RESULT_PREFIX):] for line in out.splitlines() if line.startswith(RESULT_PREFIX)]
    if exit_code != 0 or not results:
        raise AirflowException(f"« {' '.join(module_args)} » en échec (code {exit_code})")
    return json.loads(results[-1])


def check_trigger(**context) -> bool:
    """Assez de nouvelles images validées (ou un challenger à juger) ? Sinon, le DAG s'arrête ici."""
    decision = run_in_engine("src.mlops.continuous", "--check")
    context["ti"].xcom_push(key="decision", value=decision)
    print(f"Décision : {decision['reason']}")
    return bool(decision["run"])


def retrain_cycle(**context) -> None:
    """Cycle complet : entraînement → challenger → promotion (garde-fous inchangés)."""
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
    catchup=False,                   # pas de rattrapage des jours passés
    max_active_runs=1,               # un seul cycle à la fois (en plus du verrou du cycle)
    dagrun_timeout=timedelta(hours=2),
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["alpr", "mlops", "retraining"],
) as dag:
    check = ShortCircuitOperator(task_id="check_trigger", python_callable=check_trigger)
    cycle = PythonOperator(task_id="retrain_cycle", python_callable=retrain_cycle)
    summary = PythonOperator(task_id="report", python_callable=report)

    check >> cycle >> summary
