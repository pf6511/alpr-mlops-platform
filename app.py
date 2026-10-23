"""
ALPR Engine - point d'entrée de l'interface web.

    python app.py

Ce fichier ne fait qu'assembler et démarrer :
    1. build_services() crée les composants (base, moteur d'inférence, services)
    2. build_app()      construit l'interface Gradio à partir de ces services
    3. launch()         démarre le serveur web

La logique métier est dans src/services, l'interface dans src/ui,
l'inférence dans src/inference.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.services import build_services  # noqa: E402
from src.ui.build import build_app, launch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
os.makedirs("outputs", exist_ok=True)

services = build_services()
demo = build_app(services)

if __name__ == "__main__":
    launch(demo, services)
