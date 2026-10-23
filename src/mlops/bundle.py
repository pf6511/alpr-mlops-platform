"""
Écriture d'un paquet de modèle de marque (le format lu par src/inference).

    weights.pt      poids seuls (state_dict) : chargeables sans exécuter de code
    classes.json    classes, dans l'ordre des sorties du réseau
    manifest.json   lignage : jeu de données, entraînement, métriques
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

from src.inference.loader import (
    BRAND_ARCHITECTURE,
    BRAND_INPUT_SIZE,
    BUNDLE_CLASSES,
    BUNDLE_MANIFEST,
    BUNDLE_WEIGHTS,
)


def write_bundle(directory: Path, state_dict, classes: Sequence[str], manifest: Dict[str, Any]) -> Path:
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in state_dict.items()}, directory / BUNDLE_WEIGHTS)
    (directory / BUNDLE_CLASSES).write_text(json.dumps(list(classes), ensure_ascii=False, indent=2), encoding="utf-8")
    full_manifest = {
        "architecture": BRAND_ARCHITECTURE,
        "input_size": BRAND_INPUT_SIZE,
        "num_classes": len(classes),
        **manifest,
    }
    (directory / BUNDLE_MANIFEST).write_text(
        json.dumps(full_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return directory
