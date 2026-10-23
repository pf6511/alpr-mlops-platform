"""Métriques de classification (numpy seul : testables sans torch)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def classification_metrics(probs: np.ndarray, targets: Sequence[int], classes: Sequence[str]) -> Dict:
    """
    probs   : (n, nb_classes) probabilités prédites
    targets : (n,) indices des vraies classes

    → accuracy, top3_accuracy, macro_f1 (sur les classes présentes dans le test),
      et le rappel par classe.
    """
    targets = np.asarray(targets, dtype=int)
    if len(targets) == 0:
        return {"n": 0, "accuracy": 0.0, "top3_accuracy": 0.0, "macro_f1": 0.0, "per_class_recall": {}}

    probs = np.asarray(probs, dtype=float)
    predicted = probs.argmax(axis=1)
    top3 = np.argsort(-probs, axis=1)[:, :3]

    f1_scores, recalls = [], {}
    for c in sorted(set(targets.tolist())):
        tp = int(((predicted == c) & (targets == c)).sum())
        fp = int(((predicted == c) & (targets != c)).sum())
        fn = int(((predicted != c) & (targets == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls[classes[c]] = recall

    return {
        "n": int(len(targets)),
        "accuracy": float((predicted == targets).mean()),
        "top3_accuracy": float(np.any(top3 == targets[:, None], axis=1).mean()),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class_recall": recalls,
    }
