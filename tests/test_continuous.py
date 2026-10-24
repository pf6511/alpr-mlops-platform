"""
Réentraînement automatique : déclenchement, enchaînement entraînement → promotion,
mémoire dans le registre, reprise d'une promotion interrompue, verrou, rechargement
du champion par l'application. Sans torch ni MLflow (entraînement et promotion simulés).
"""

import copy
import json
from types import SimpleNamespace

from configs.settings import get_settings
from src.mlops import continuous
from src.mlops.registry import PROMOTION_TAG, VALIDATED_IMAGES_TAG, RegisteredBrandModel


class FakeStorage:
    def __init__(self, labels):
        self.keys = [f"labeled/{label}/{i:03d}.jpg" for i, label in enumerate(labels)]

    def list_files(self, prefix="", bucket_type="dataset", max_keys=1000):
        return [SimpleNamespace(key=k) for k in self.keys if k.startswith(prefix)]


class FakeRegistry:
    """Versions, alias et tags en mémoire : le même rôle que le registre MLflow."""

    def __init__(self):
        self.models, self.aliases = {}, {}

    def add(self, tags=None, alias=None):
        version = str(len(self.models) + 1)
        self.models[version] = RegisteredBrandModel(version, f"run{version}", "src", tags=dict(tags or {}))
        if alias:
            self.aliases[alias] = version
        return self.models[version]

    def versions(self):
        return list(self.models.values())

    def get(self, alias):
        return self.models.get(self.aliases.get(alias))


def settings(min_new_images=5):
    s = copy.deepcopy(get_settings())
    s.continuous.min_new_images = min_new_images
    return s


def fake_train(calls, registry, n_validated):
    def train(settings, storage, reg, **kwargs):
        calls.append(("train", kwargs))
        version = registry.add(tags={"origin": "training", VALIDATED_IMAGES_TAG: str(n_validated)},
                               alias="challenger")
        return {"version": version.version}
    return train


def fake_promote(calls, registry, promote=True):
    def promote_fn(settings, storage, reg, min_test_images):
        calls.append(("promote", min_test_images))
        challenger = registry.get("challenger")
        challenger.tags[PROMOTION_TAG] = "promoted" if promote else "rejected"
        if promote:
            registry.aliases["champion"] = challenger.version
        return {"promoted": promote, "reason": "règle", "challenger_version": challenger.version}
    return promote_fn


def bootstrapped_registry():
    registry = FakeRegistry()
    registry.add(tags={"origin": "bootstrap"}, alias="champion")       # v1, aucune image validée
    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# DÉCLENCHEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def test_trigger_rule():
    assert continuous.decide_trigger(12, 10, 5).run is False
    assert continuous.decide_trigger(15, 10, 5).run is True
    assert continuous.decide_trigger(10, 10, 5, force=True).run is True
    assert continuous.decide_trigger(15, 10, 5).new_images == 5


def test_only_known_brands_are_counted():
    storage = FakeStorage(["renault", "peugeot", "tesla"])               # tesla : hors classes
    assert continuous.count_validated(storage, get_settings().models.brand_classes) == 2


def test_registry_remembers_last_training():
    registry = bootstrapped_registry()
    assert continuous.last_trained_count(registry) == 0
    registry.add(tags={VALIDATED_IMAGES_TAG: "40"})
    registry.add(tags={VALIDATED_IMAGES_TAG: "25"})
    assert continuous.last_trained_count(registry) == 40


# ═══════════════════════════════════════════════════════════════════════════════
# CYCLE
# ═══════════════════════════════════════════════════════════════════════════════

def test_not_enough_new_images_skips():
    registry, calls = bootstrapped_registry(), []
    report = continuous.run_cycle(settings(), FakeStorage(["renault"] * 3), registry,
                                  train_fn=fake_train(calls, registry, 3), promote_fn=fake_promote(calls, registry))
    assert report["status"] == "skipped" and calls == []


def test_cycle_trains_then_promotes_and_is_idempotent():
    registry, calls = bootstrapped_registry(), []
    storage = FakeStorage(["renault"] * 4 + ["peugeot"] * 4)
    train, promote = fake_train(calls, registry, 8), fake_promote(calls, registry)

    report = continuous.run_cycle(settings(), storage, registry, train_fn=train, promote_fn=promote)
    assert report["status"] == "promoted" and [c[0] for c in calls] == ["train", "promote"]
    assert registry.get("champion").version == "2"
    assert report["trigger"]["new_images"] == 8

    # Relancé sans nouvelles images : rien à faire
    again = continuous.run_cycle(settings(), storage, registry, train_fn=train, promote_fn=promote)
    assert again["status"] == "skipped" and len(calls) == 2


def test_rejected_challenger_keeps_champion():
    registry, calls = bootstrapped_registry(), []
    report = continuous.run_cycle(settings(), FakeStorage(["renault"] * 8), registry,
                                  train_fn=fake_train(calls, registry, 8),
                                  promote_fn=fake_promote(calls, registry, promote=False))
    assert report["status"] == "not_promoted" and registry.get("champion").version == "1"


def test_interrupted_promotion_is_resumed_before_training():
    registry, calls = bootstrapped_registry(), []
    registry.add(tags={VALIDATED_IMAGES_TAG: "8"}, alias="challenger")  # entraîné, jamais jugé
    report = continuous.run_cycle(settings(), FakeStorage(["renault"] * 30), registry,
                                  train_fn=fake_train(calls, registry, 30), promote_fn=fake_promote(calls, registry))
    assert [c[0] for c in calls] == ["promote"] and report["status"] == "promoted"


def test_guarded_cycle_records_errors_and_respects_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(continuous, "HISTORY_FILE", tmp_path / "history.jsonl")
    monkeypatch.setattr(continuous, "LOCK_FILE", tmp_path / "retrain.lock")

    def broken(**kwargs):
        raise ConnectionError("registre injoignable")
    monkeypatch.setattr(continuous, "run_cycle", broken)

    report = continuous.guarded_cycle()
    assert report["status"] == "error" and "injoignable" in report["reason"]
    assert json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])["status"] == "error"

    with continuous.cycle_lock(tmp_path / "retrain.lock") as first:
        with continuous.cycle_lock(tmp_path / "retrain.lock") as second:
            assert first is True and second is False


# ═══════════════════════════════════════════════════════════════════════════════
# L'APPLICATION CHARGE LE NOUVEAU CHAMPION
# ═══════════════════════════════════════════════════════════════════════════════

def test_application_reloads_only_when_champion_changes():
    from src.services.system import SystemService

    registry, loaded = bootstrapped_registry(), []
    engine = SimpleNamespace(models=SimpleNamespace(sources={"brand_classifier": "registry:m@champion (v1)"}))

    def replace(ref):
        loaded.append(ref.label)
        engine.models.sources["brand_classifier"] = ref.label
        return True, f"✅ {ref.label}"
    engine.replace_brand_classifier = replace
    registry.download = lambda alias=None: SimpleNamespace(
        label=f"registry:m@champion (v{registry.get('champion').version})")

    system = SystemService(engine, get_settings(), registry=registry)
    assert system.refresh_champion_if_changed()[0] is False and loaded == []     # v1 déjà servie

    registry.add(alias="champion")                                               # promotion de v2
    assert system.refresh_champion_if_changed()[0] is True
    assert loaded == ["registry:m@champion (v2)"]
    assert system.refresh_champion_if_changed()[0] is False                      # pas de rechargement inutile
    assert SystemService(engine, get_settings(), registry=None).start_champion_watch(10) is None


def test_validation_tab_shows_progress_towards_retraining():
    from src.services.validation import ValidationService
    from src.ui.formatters import dataset_stats

    registry, storage = bootstrapped_registry(), FakeStorage(["renault"] * 7)
    progress = continuous.retraining_progress(settings(min_new_images=5), storage, registry)
    assert progress == {"new_images": 7, "threshold": 5, "validated_images": 7, "will_retrain": True}

    html = dataset_stats(ValidationService(None, lambda: progress).stats())
    assert "7 / 5" in html and "prochain cycle" in html
