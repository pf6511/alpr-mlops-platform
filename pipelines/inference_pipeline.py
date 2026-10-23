"""
ALPR Engine - Adaptateur ligne de commande du cœur d'inférence.

    python -m pipelines.inference_pipeline image.jpg --declared-brand Renault

Toute l'inférence est déléguée à `src.inference.ALPREngine` : ce module ne fait
que lire les arguments et mettre le résultat au format JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import ALPREngine, is_brand_mismatch  # noqa: E402


class InferencePipeline:
    """Résultats d'inférence au format JSON (CLI, scripts, traitements par lot)."""

    def __init__(self, load_all: bool = True, engine: Optional[ALPREngine] = None):
        self.engine = engine or ALPREngine(load_branch2=load_all)

    def process(self, image: Union[str, np.ndarray],
                conf_threshold: float = None,
                run_brand: bool = True,
                declared_brand: str = None) -> Dict:
        """Analyse une image et renvoie un dictionnaire sérialisable en JSON."""
        try:
            result = self.engine.analyze(image, conf_threshold=conf_threshold, run_branch2=run_brand)
        except ValueError as exc:
            return {"error": str(exc)}

        output = result.to_dict()
        predicted = result.brand.brand if result.brand else None
        output["declared_brand"] = declared_brand
        output["mismatch"] = is_brand_mismatch(predicted, declared_brand)
        return output

    def process_batch(self, images: List[Union[str, np.ndarray]], **kwargs) -> List[Dict]:
        return [self.process(img, **kwargs) for img in images]


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json
    import logging

    parser = argparse.ArgumentParser(description="ALPR - analyse d'une ou plusieurs images")
    parser.add_argument("images", nargs="+", help="Chemin(s) des images")
    parser.add_argument("--no-brand", action="store_true", help="Désactiver la classification de marque")
    parser.add_argument("--declared-brand", help="Marque déclarée, pour la détection d'incohérence")
    parser.add_argument("--conf", type=float, default=None,
                        help="Seuil de confiance des plaques (défaut : configuration)")
    parser.add_argument("--verbose", action="store_true", help="Afficher le chargement des modèles")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    pipeline = InferencePipeline(load_all=not args.no_brand)
    results = pipeline.process_batch(
        args.images,
        conf_threshold=args.conf,
        run_brand=not args.no_brand,
        declared_brand=args.declared_brand,
    )
    payload = results[0] if len(results) == 1 else results
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if all("error" not in r for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
