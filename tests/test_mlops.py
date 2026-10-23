"""
Tests de la logique MLOps sans torch ni MLflow : jeu de données et manifeste,
métriques, règle de promotion, registre (faux client), câblage du registre.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.mlops.promote import decide_promotion
from src.mlops.registry import BrandModelRegistry
from src.training.dataset import Sample, assign_split, build_manifest, collect_base, split_samples
from src.training.metrics import classification_metrics

CLASSES = ["peugeot", "renault", "volkswagen"]


# ═══════════════════════════════════════════════════════════════════════════════
# JEU DE DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

def test_split_is_stable_and_proportional():
    ids = [f"labeled/renault/{i}.jpg" for i in range(2000)]
    first = [assign_split(i) for i in ids]
    assert first == [assign_split(i) for i in ids]                 # déterministe
    assert 0.15 < first.count("test") / len(ids) < 0.25            # ≈ 20 %


def test_unknown_labels_are_rejected_and_reported():
    samples = [Sample("labeled/renault/a.jpg", "renault", "validated"),
               Sample("labeled/tesla/b.jpg", "tesla", "validated")]
    kept, rejected = split_samples(samples, CLASSES)
    assert [s.label for s in kept] == ["renault"] and [s.label for s in rejected] == ["tesla"]

    kept[0].sha256 = "abc"
    manifest = build_manifest(kept, rejected, CLASSES, {"validated": "alpr-dataset/labeled/"}, 0.2)
    assert manifest["rejected_unknown_labels"] == {"tesla": 1}
    assert manifest["counts"]["train"] + manifest["counts"]["test"] == 1
    assert {"dataset_id", "sources", "test_set", "classes"} <= set(manifest)


def test_test_set_is_listed_in_manifest():
    samples = [Sample(f"labeled/peugeot/{i}.jpg", "peugeot", "validated", sha256=str(i)) for i in range(50)]
    kept, _ = split_samples(samples, CLASSES)
    manifest = build_manifest(kept, [], CLASSES, {}, 0.2)
    test_ids = {e["id"] for e in manifest["test_set"]}
    assert test_ids == {s.identifier for s in kept if s.split == "test"}
    assert manifest["counts"]["test"] == len(test_ids) > 0


def test_base_dataset_is_collected_by_brand_folder(tmp_path):
    for brand, n in (("Renault", 3), ("VW", 2)):
        (tmp_path / brand).mkdir()
        for i in range(n):
            (tmp_path / brand / f"{i}.jpg").write_bytes(b"x")
    samples = collect_base(tmp_path, max_per_class=2)
    assert sorted(s.label for s in samples) == ["renault", "renault", "volkswagen", "volkswagen"]


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES ET PROMOTION
# ═══════════════════════════════════════════════════════════════════════════════

def test_classification_metrics():
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.5, 0.4, 0.1], [0.1, 0.2, 0.7]])
    m = classification_metrics(probs, [0, 1, 1, 2], CLASSES)
    assert m["n"] == 4 and m["accuracy"] == pytest.approx(0.75)
    assert m["top3_accuracy"] == 1.0
    assert m["per_class_recall"]["renault"] == pytest.approx(0.5)
    assert classification_metrics(np.zeros((0, 3)), [], CLASSES)["n"] == 0


def test_promotion_rule():
    champion = {"n": 50, "accuracy": 0.80, "macro_f1": 0.75}
    assert decide_promotion(champion, {"n": 50, "accuracy": 0.82, "macro_f1": 0.76}).promote is True
    assert decide_promotion(champion, {"n": 50, "accuracy": 0.79, "macro_f1": 0.90}).promote is False
    assert decide_promotion(champion, {"n": 50, "accuracy": 0.85, "macro_f1": 0.70}).promote is False
    assert decide_promotion(champion, {"n": 50, "accuracy": 0.85, "macro_f1": 0.70}, f1_tolerance=0.1).promote
    small = decide_promotion(champion, {"n": 5, "accuracy": 1.0, "macro_f1": 1.0})
    assert small.promote is False and "trop petit" in small.reason


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRE (faux client MLflow)
# ═══════════════════════════════════════════════════════════════════════════════

class FakeMlflow:
    """Imite les appels de mlflow utilisés par le registre."""

    def __init__(self, root: Path):
        self.root, self.runs, self.experiment = root, 0, None
        self.artifacts = SimpleNamespace(download_artifacts=self._download)

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name=None):
        self.runs += 1
        run_dir = self.root / f"run{self.runs}"
        run = SimpleNamespace(info=SimpleNamespace(run_id=f"run{self.runs}", artifact_uri=str(run_dir)))
        mlflow = self

        class _Ctx:
            def __enter__(self):
                mlflow.current = run_dir
                return run

            def __exit__(self, *a):
                return False
        return _Ctx()

    def log_params(self, p): self.params = p
    def log_metrics(self, m, step=None): self.metrics = m
    def set_tags(self, t): self.tags = t

    def log_artifacts(self, local_dir, artifact_path):
        shutil.copytree(local_dir, self.current / artifact_path)

    def _download(self, artifact_uri, dst_path):
        target = Path(dst_path) / Path(artifact_uri).name
        shutil.copytree(artifact_uri, target)
        return str(target)


class FakeClient:
    def __init__(self):
        self.versions, self.aliases, self.registered, self.tags = {}, {}, False, {}

    def get_registered_model(self, name):
        if not self.registered:
            raise LookupError(name)

    def create_registered_model(self, name, description=None):
        self.registered = True

    def create_model_version(self, name, source, run_id, tags=None):
        version = str(len(self.versions) + 1)
        self.versions[version] = SimpleNamespace(version=version, source=source, run_id=run_id,
                                                 status="READY", aliases=[], tags=tags or {})
        return self.versions[version]

    def get_model_version(self, name, version):
        return self.versions[version]

    def get_model_version_by_alias(self, name, alias):
        return self.versions[self.aliases[alias]]

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(version, key)] = value


def make_registry(tmp_path):
    from configs.settings import MLflowSettings

    cfg = MLflowSettings(cache_dir=str(tmp_path / "cache"))
    return BrandModelRegistry(cfg, client=FakeClient(), mlflow_module=FakeMlflow(tmp_path / "store"))


def fake_bundle(tmp_path, name="bundle"):
    d = tmp_path / name
    d.mkdir()
    (d / "weights.pt").write_bytes(b"poids")
    (d / "classes.json").write_text(json.dumps(CLASSES))
    return d


def test_registry_register_alias_and_download(tmp_path):
    registry = make_registry(tmp_path)
    assert registry.get("champion") is None

    v1 = registry.register(fake_bundle(tmp_path), run_name="r", params={}, metrics={}, tags={}, alias="champion")
    assert v1.version == "1" and registry.get("champion").version == "1"

    ref = registry.download()                                     # champion par défaut
    assert (ref.directory / "weights.pt").read_bytes() == b"poids"
    assert ref.label == "registry:vehicle_brand_model@champion (v1)"
    assert ref.directory == registry.download().directory         # cache réutilisé

    registry.register(fake_bundle(tmp_path, "b2"), run_name="r2", params={}, metrics={}, tags={}, alias="challenger")
    assert registry.get("challenger").version == "2" and registry.get("champion").version == "1"
    registry.set_alias("champion", "2")
    assert registry.download().label.endswith("(v2)")


# ═══════════════════════════════════════════════════════════════════════════════
# CÂBLAGE DANS L'APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_unreachable_registry_falls_back_to_local(monkeypatch):
    from configs.settings import get_settings
    from src.services import container

    settings = get_settings()
    monkeypatch.setattr(settings.mlflow, "brand_model_source", "registry")
    monkeypatch.setattr(BrandModelRegistry, "download", lambda self, alias=None: (_ for _ in ()).throw(OSError("injoignable")))
    registry, bundle = container._registry_and_champion(settings)
    assert registry is not None and bundle is None                # l'application démarre quand même


def test_reload_champion_from_settings_tab():
    from src.services.system import SystemService

    calls = []
    engine = SimpleNamespace(replace_brand_classifier=lambda ref: (calls.append(ref) or (True, f"✅ {ref.label}")))
    registry = SimpleNamespace(download=lambda: SimpleNamespace(label="registry:m@champion (v3)"))

    assert SystemService(engine, None, registry=registry).reload_brand_champion() == (True, "✅ registry:m@champion (v3)")
    assert SystemService(engine, None, registry=None).reload_brand_champion()[0] is False
