"""
Tests du monitoring (src/monitoring/report.py).

Passages synthétiques, fausse base : on vérifie les périodes, les indicateurs,
les seuils et l'enchaînement complet. Le rapport Evidently n'est testé que si
la bibliothèque est installée.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta

import pandas as pd
import pytest

from configs.settings import MonitoringSettings
from src.monitoring import report as monitoring

NOW = datetime(2026, 11, 10, 12, 0)
SETTINGS = MonitoringSettings(current_days=7, reference_days=30, min_passages=5,
                              max_brand_error_rate=0.20, min_brand_confidence=0.70,
                              max_duration_ms_p95=3000)


def passage(days_ago, **overrides):
    row = {
        "id": 1, "timestamp": NOW - timedelta(days=days_ago), "source": "interface",
        "motif": "AUTHORIZED", "resultat": "autorisé", "marque_predite": "renault",
        "marque_confiance": 0.9, "mismatch": False, "validation_status": None,
        "validated_label": None, "ocr_confiance": 0.95, "luminosite": 120.0,
        "nettete": 300.0, "duree_ms": 400, "modele_marque": "champion v1",
    }
    return row | overrides


def frame(rows):
    return pd.DataFrame(rows)


class FakeDB:
    """Imite DatabaseManager.get_cursor() : renvoie les lignes postérieures à `since`."""

    def __init__(self, rows):
        self.rows = rows

    @contextmanager
    def get_cursor(self):
        rows = self.rows

        class Cursor:
            def execute(self, sql, params):
                assert "monitoring.passages" in sql            # on lit la vue, pas la table
                self.result = [r for r in rows if r["timestamp"] >= params[0]]

            def fetchall(self):
                return self.result

        yield Cursor()


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(monitoring, "HISTORY_FILE", tmp_path / "history.jsonl")


# ═══════════════════════════════════════════════════════════════════════════════
# PÉRIODES ET INDICATEURS
# ═══════════════════════════════════════════════════════════════════════════════

def test_split_periods():
    df = frame([passage(1), passage(6), passage(10), passage(36), passage(40)])
    reference, current = monitoring.split_periods(df, NOW, SETTINGS)
    assert len(current) == 2          # moins de 7 jours
    assert len(reference) == 2        # entre 7 et 37 jours ; celui de 40 jours est hors fenêtre


def test_indicators():
    df = frame([
        passage(1, duree_ms=300),
        passage(1, resultat="refusé", motif="UNKNOWN_PLATE", duree_ms=500),
        passage(1, mismatch=True, marque_confiance=0.6, duree_ms=700),
        passage(1, mismatch=True, marque_confiance=0.5, duree_ms=900),
    ])
    ind = monitoring.compute_indicators(df)
    assert ind["passages"] == 4
    assert ind["part_refus"] == 0.25
    assert ind["part_incoherences"] == 0.5
    assert ind["confiance_marque_moyenne"] == 0.725
    assert ind["taux_erreur_marque"] is None and ind["validations"] == 0   # pas encore de validation


def test_brand_error_rate_from_operator_validations():
    df = frame([
        # le modèle avait raison : l'opérateur retient la marque prédite (vraie usurpation)
        passage(1, mismatch=True, validation_status="validated", marque_predite="audi", validated_label="Audi"),
        # le modèle s'est trompé : l'opérateur retient une autre marque
        passage(1, mismatch=True, validation_status="validated", marque_predite="audi", validated_label="volkswagen"),
        # en attente : pas compté
        passage(1, mismatch=True, validation_status=None, marque_predite="bmw"),
    ])
    rate, count = monitoring.brand_error_rate(df)
    assert count == 2 and rate == 0.5


def test_empty_period():
    assert monitoring.compute_indicators(frame([])) == {"passages": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# SEUILS
# ═══════════════════════════════════════════════════════════════════════════════

def test_thresholds_ok():
    ind = {"taux_erreur_marque": 0.1, "confiance_marque_moyenne": 0.85, "duree_ms_p95": 800}
    assert monitoring.check_thresholds(ind, SETTINGS) == []


def test_thresholds_alerts():
    ind = {"taux_erreur_marque": 0.35, "confiance_marque_moyenne": 0.55, "duree_ms_p95": 4200}
    alerts = monitoring.check_thresholds(ind, SETTINGS)
    assert len(alerts) == 3 and "35%" in alerts[0]


def test_thresholds_not_measurable():
    """Pas de validation, pas de passage : rien à contrôler, pas d'alerte."""
    assert monitoring.check_thresholds({"passages": 0}, SETTINGS) == []


# ═══════════════════════════════════════════════════════════════════════════════
# ENCHAÎNEMENT COMPLET
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_without_drift_writes_history():
    rows = [passage(d) for d in (1, 2, 3)] + [passage(d) for d in (10, 20)]
    result = monitoring.run(now=NOW, with_drift=False, db=FakeDB(rows), settings=SETTINGS)
    assert result["statut"] == "ok"
    assert result["recente"]["passages"] == 3 and result["reference"]["passages"] == 2
    assert monitoring.HISTORY_FILE.read_text(encoding="utf-8").count("\n") == 1   # une ligne d'historique


def test_run_skips_drift_when_too_few_passages():
    result = monitoring.run(now=NOW, db=FakeDB([passage(1)]), settings=SETTINGS)
    assert "ignore" in result["derive"] and result["rapport_html"] is None


def test_run_raises_alert():
    rows = [passage(1, marque_confiance=0.4) for _ in range(3)]
    result = monitoring.run(now=NOW, with_drift=False, db=FakeDB(rows), settings=SETTINGS)
    assert result["statut"] == "alerte" and result["alertes"]


def test_drifted_share_reads_evidently_result():
    result = {"metrics": [{"metric_id": "DriftedColumnsCount(drift_share=0.5)",
                           "value": {"count": 2, "share": 0.25}}]}
    assert monitoring.drifted_share(result) == 0.25
    other_layout = {"metrics": [{"config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                                 "value": {"count": 1, "share": 0.125}}]}
    assert monitoring.drifted_share(other_layout) == 0.125
    assert monitoring.drifted_share({"metrics": []}) is None


def test_drift_report_with_evidently():
    pytest.importorskip("evidently")
    reference = [passage(10 + i % 20, luminosite=120.0 + i % 5) for i in range(40)]
    current = [passage(1 + i % 5, luminosite=40.0 + i % 5, marque_predite="audi") for i in range(40)]
    result = monitoring.run(now=NOW, db=FakeDB(reference + current), settings=SETTINGS)
    assert result["rapport_html"] and monitoring.Path(result["rapport_html"]).exists()
    assert result["derive"]["part_colonnes_en_derive"] > 0      # luminosité et marque ont changé
