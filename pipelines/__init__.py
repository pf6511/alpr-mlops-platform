"""
ALPR Engine - Pipelines module.

Les classes sont importées à la demande (PEP 562) : lancer la CLI avec
`python -m pipelines.inference_pipeline` ne charge ni le module deux fois,
ni le code d'entraînement.
"""

from importlib import import_module

_EXPORTS = {
    "InferencePipeline": ".inference_pipeline",
    "TrainingPipeline": ".training_pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
