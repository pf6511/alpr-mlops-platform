"""
Fonctions commun aux DAG ALPR : exécuter une commande du projet DANS le conteneur alpr-engine.

- Airflow décide QUAND 
- le projet décide QUOI et SI
- Les DAG ne contiennent aucune règle métier : ils lancent « python -m <module> » dans l'application (via l'API Docker) et
lisent sa dernière ligne de sortie « ALPR_RESULT {json} ».

Ce fichier ne définit pas de DAG : Airflow l'ignore, les DAG l'importent (le dossier
des DAG est dans le chemin d'import de Python).

Remarques : Airflow exécute les commandes du projet dans le conteneur de l'application, 
grâce au SDK Docker et au socket Docker monté dans son conteneur.

Ce choix a été motivé et même plutôt contraint par les mimites matérielles : la mémoire sur la machine.
C'est simple et sans duplication de l'environnement, mais l"accès au socket équivaut à des droits d'administrateur. 
En production, je le remplacerais par un opérateur qui lance un conteneur dédié par tâche.
Idéalement Airflow découpé en composants (planificateur, interface…) KubernetesPOdOperator ou Kuberneted Executot : chaque tâche
dans un pod éphémère...

"""

from __future__ import annotations

import json
import os

from airflow.exceptions import AirflowException

ENGINE_CONTAINER = os.environ.get("ALPR_ENGINE_CONTAINER", "alpr-engine")
RESULT_PREFIX = "ALPR_RESULT "          # dernière ligne de sortie des commandes du projet


def run_in_engine(*module_args: str) -> dict:
    """Exécute « python -m <module> <args> » dans le conteneur de l'application et renvoie son résultat."""
    import docker
    # _PIP_ADDITIONAL_REQUIREMENTS=docker>=6.1
    container = docker.from_env().containers.get(ENGINE_CONTAINER)
    exit_code, (stdout, stderr) = container.exec_run(
        ["python", "-m", *module_args], workdir="/app", demux=True
    )
    out = (stdout or b"").decode("utf-8", "replace")
    err = (stderr or b"").decode("utf-8", "replace")
    print(out)
    if err:
        print(err)                           # logs de la commande

    results = [line[len(RESULT_PREFIX):] for line in out.splitlines() if line.startswith(RESULT_PREFIX)]
    if exit_code != 0 or not results:
        raise AirflowException(f"« {' '.join(module_args)} » en échec (code {exit_code})")
    return json.loads(results[-1])
