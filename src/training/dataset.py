"""
Construction du jeu de données d'entraînement de marque, avec son manifeste.

Sources :
- les images validées par les opérateurs : bucket « dataset », préfixe labeled/<marque>/
- un socle facultatif (ex. VMMRdb regroupé par marque) : dossier local <marque>/*.jpg

Répartition entraînement / test par empreinte de l'identifiant de chaque image :
une image tombe toujours du même côté. Le jeu de test reste donc stable quand de
nouvelles images arrivent, ce qui rend les comparaisons entre modèles honnêtes.
"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.inference.brands import normalize_brand

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LABELED_PREFIX = "labeled/"


@dataclass
class Sample:
    identifier: str        # clé du bucket (« labeled/peugeot/xxx.jpg ») ou chemin du socle
    label: str             # marque normalisée
    origin: str            # « validated » ou « base »
    split: str = ""        # « train » ou « test »
    local_path: Optional[Path] = None
    sha256: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTE
# ═══════════════════════════════════════════════════════════════════════════════

def collect_validated(storage) -> List[Sample]:
    """Images validées en production : labeled/<marque>/<fichier> dans le bucket dataset."""
    samples = []
    for f in storage.list_files(prefix=LABELED_PREFIX, bucket_type="dataset", max_keys=100_000):
        parts = f.key.replace("\\", "/").split("/")
        if len(parts) >= 3 and Path(f.key).suffix.lower() in IMAGE_SUFFIXES:
            samples.append(Sample(identifier=f.key.replace("\\", "/"), label=normalize_brand(parts[1]),
                                  origin="validated"))
    return samples


def collect_base(base_dir: Optional[Path], max_per_class: Optional[int] = None) -> List[Sample]:
    """Socle local : <base_dir>/<marque>/*.jpg (tri par nom, pour un échantillon reproductible)."""
    if not base_dir:
        return []
    samples = []
    for class_dir in sorted(p for p in Path(base_dir).iterdir() if p.is_dir()):
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for path in files[:max_per_class] if max_per_class else files:
            samples.append(Sample(identifier=str(path), label=normalize_brand(class_dir.name),
                                  origin="base", local_path=path))
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# RÉPARTITION
# ═══════════════════════════════════════════════════════════════════════════════

def assign_split(identifier: str, test_fraction: float = 0.2) -> str:
    """« test » ou « train », de façon déterministe à partir de l'identifiant."""
    bucket = int(hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if bucket < test_fraction else "train"


def split_samples(samples: Iterable[Sample], classes: Sequence[str], test_fraction: float = 0.2):
    """Attribue les répartitions ; écarte les marques inconnues du modèle. → (retenus, écartés)"""
    known = set(classes)
    kept, rejected = [], []
    for sample in samples:
        if sample.label not in known:
            rejected.append(sample)
            continue
        sample.split = assign_split(sample.identifier, test_fraction)
        kept.append(sample)
    return kept, rejected


# ═══════════════════════════════════════════════════════════════════════════════
# MATÉRIALISATION ET MANIFESTE
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def materialize(samples: List[Sample], storage, work_dir: Path) -> List[Sample]:
    """Copie chaque image dans work_dir/<split>/<marque>/ et calcule son empreinte."""
    ready = []
    for i, sample in enumerate(samples):
        dest = Path(work_dir) / sample.split / sample.label / f"{i:06d}{Path(sample.identifier).suffix.lower()}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if sample.origin == "validated":
            ok, _ = storage.download_file(sample.identifier, dest, bucket_type="dataset")
            if not ok:
                continue
        else:
            shutil.copyfile(sample.local_path, dest)
        sample.local_path = dest
        sample.sha256 = _sha256(dest)
        ready.append(sample)
    return ready


def build_manifest(samples: List[Sample], rejected: List[Sample], classes: Sequence[str],
                   sources: Dict[str, str], test_fraction: float) -> Dict:
    """Manifeste de lignage : d'où viennent les données, combien, et lesquelles servent au test."""
    by_split = {split: Counter(s.label for s in samples if s.split == split) for split in ("train", "test")}
    fingerprint = hashlib.sha256("".join(sorted(s.sha256 for s in samples)).encode()).hexdigest()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": fingerprint[:16],
        "sources": sources,
        "test_fraction": test_fraction,
        "classes": list(classes),
        "counts": {
            "train": sum(by_split["train"].values()),
            "test": sum(by_split["test"].values()),
            "by_origin": dict(Counter(s.origin for s in samples)),
            "train_per_class": dict(sorted(by_split["train"].items())),
            "test_per_class": dict(sorted(by_split["test"].items())),
        },
        "rejected_unknown_labels": dict(Counter(s.label for s in rejected)),
        # Le jeu de test est listé explicitement : la promotion réévalue les deux modèles dessus
        "test_set": [{"id": s.identifier, "label": s.label, "origin": s.origin, "sha256": s.sha256}
                     for s in samples if s.split == "test"],
    }
