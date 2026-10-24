"""
Mise en forme des résultats pour l'interface (HTML, Markdown, tableaux, images).

Fonctions pures : elles reçoivent des objets métier et renvoient de quoi
afficher. Aucune logique métier, aucun accès à la base.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import pandas as pd

from src.inference import BrandResult, InferenceResult, VehicleResult
from src.inference.rendering import annotate, draw_plate_boxes
from src.utils.visualizer import create_confidence_badge, format_ocr_result

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def to_rgb(image_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image_bgr is None or image_bgr.ndim != 3:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def format_timestamp(value: Any) -> str:
    """Horodatage lisible, qu'il vienne de PostgreSQL (datetime) ou de SQLite (texte)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value).replace("T", " ")[:19]


def _percent(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1%}"


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPES DU PIPELINE (images)
# ═══════════════════════════════════════════════════════════════════════════════

def pipeline_images(image_bgr: np.ndarray, result: InferenceResult, mismatch: bool):
    """Les quatre images de l'onglet : brute, détection, plaque recadrée, finale (RGB)."""
    rgb = to_rgb(image_bgr)
    first_crop = to_rgb(result.plates[0].crop) if result.plates else None
    return rgb, draw_plate_boxes(rgb, result), first_crop, annotate(rgb, result, mismatch=mismatch)


# ═══════════════════════════════════════════════════════════════════════════════
# CARTES HTML
# ═══════════════════════════════════════════════════════════════════════════════

def brand_card(brand: Optional[BrandResult], mismatch: bool, declared_brand: Optional[str]) -> str:
    if brand is None:
        return "<p style='color: #888;'>🚗 Marque: Non détectée</p>"

    if mismatch:
        color, icon, status = "#ef4444", "⚠️", f"MISMATCH (déclarée: {declared_brand})"
    else:
        color, icon, status = "#22c55e", "✅", "OK"

    top3 = "".join(f"<li>{name}: {conf:.1%}</li>" for name, conf in brand.top3[:3])
    return f"""
    <div style="background: linear-gradient(135deg, {color}22, {color}11);
                border: 2px solid {color}; border-radius: 12px; padding: 15px; margin: 10px 0;">
        <h3 style="margin: 0; color: {color};">{icon} Classification Marque: {status}</h3>
        <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">{brand.brand}</p>
        <p style="margin: 5px 0;">Confiance: <strong>{brand.confidence:.1%}</strong></p>
        <p style="font-size: 12px; color: #888;">Marque déclarée: {declared_brand or "Non déclarée"}</p>
        <hr style="border-color: {color}33;">
        <p style="font-size: 12px; color: #888;">Top 3:</p>
        <ul style="margin: 5px 0; padding-left: 20px;">{top3}</ul>
    </div>
    """


def vehicle_card(vehicle: Optional[VehicleResult]) -> str:
    if vehicle is None:
        return "<p style='color: #888;'>🚙 Véhicule: Non détecté</p>"
    return f"""
    <div style="background: #3b82f622; border: 1px solid #3b82f6;
                border-radius: 8px; padding: 10px; margin: 5px 0;">
        <p style="margin: 0;"><strong>🚙 Véhicule détecté:</strong> {vehicle.class_name.capitalize()}</p>
        <p style="margin: 5px 0; font-size: 12px;">Confiance: {vehicle.confidence:.1%}</p>
    </div>
    """


def access_banner(access) -> str:
    if access is None:
        color, text = "#64748b", "⚠️ NO PLATE DETECTED - ACCESS UNKNOWN"
    else:
        color, text = ("green" if access.authorized else "red"), access.message
    return f"""<div class="access-banner" style="background-color: {color};
        color: white; text-align: center; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
        {text}
    </div>"""


MISMATCH_WARNING = """
<div style="background: #fef2f2; border: 2px solid #ef4444;
            border-radius: 8px; padding: 15px; margin: 10px 0;">
    <h3 style="color: #dc2626; margin: 0;">⚠️ MISMATCH DÉTECTÉ</h3>
    <p>La marque prédite ne correspond pas à la marque déclarée.</p>
    <p>Ce cas a été ajouté à la file de validation.</p>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# TEXTES
# ═══════════════════════════════════════════════════════════════════════════════

def ocr_summary(result: InferenceResult) -> str:
    if not result.plates:
        return "❌ No plates detected"
    return "\n\n".join(format_ocr_result(p.text, p.ocr_confidence or 0.0) for p in result.plates)


def analysis_report(result: InferenceResult) -> str:
    """Rapport Markdown : conditions de prise de vue et détail des plaques."""
    c = result.conditions
    report = f"""
## 🔍 Analysis Report

### Image Conditions
- **Lighting**: {c.lighting_emoji} {c.lighting}
- **Brightness**: {c.brightness:.1f} / 255
- **Sharpness**: {c.blur} (score: {c.blur_score:.1f})
- **Inference time**: {result.inference_time_s:.2f} s

### Detections
"""
    if not result.plates:
        return report + "\n❌ **No plates detected**\n"

    for i, plate in enumerate(result.plates, 1):
        if plate.ocr_confidence is None:
            ocr_line = "N/A (non fournie par le moteur OCR)"
        else:
            emoji, _, label = create_confidence_badge(plate.ocr_confidence)
            ocr_line = f"{emoji} {plate.ocr_confidence:.2%} ({label})"
        report += f"""
#### Plate {i}
- **Text**: `{plate.text or 'Not detected'}`
- **OCR Confidence**: {ocr_line}
- **Detection Confidence**: {plate.detection_confidence:.2%}
"""
        if plate.ocr_confidence is not None and plate.ocr_confidence < 0.6:
            report += (f"\n> ⚠️ **Low OCR confidence** - Possible reasons: {c.lighting}, "
                       f"{c.blur}, extreme angle, or dirty plate\n")
    return report


def capture_report(outcome) -> str:
    """Bandeau d'accès + alerte d'incohérence éventuelle + rapport d'analyse."""
    parts = [access_banner(outcome.access)]
    if outcome.mismatch:
        parts.append(MISMATCH_WARNING)
    parts.append(analysis_report(outcome.result))
    return "\n\n".join(parts)


def _retraining_line(progress: Optional[Dict[str, Any]]) -> str:
    """Progression vers le prochain réentraînement automatique (même règle que le cycle)."""
    if not progress:
        return "<p>Réentraînement automatique : <strong>registre non configuré</strong></p>"
    status = "✅ au prochain cycle" if progress["will_retrain"] else "⏳ pas encore"
    return (f"<p>Nouvelles images validées depuis le dernier entraînement : "
            f"<strong>{progress['new_images']} / {progress['threshold']}</strong></p>"
            f"<p>Réentraînement automatique : <strong>{status}</strong></p>")


def dataset_stats(stats: Dict[str, Any]) -> str:
    if not stats:
        return "Dataset stats non disponibles"
    dataset = stats.get("dataset", {}) or {}
    return f"""
    <div style="background: #1e293b; padding: 20px; border-radius: 12px;">
        <h3 style="color: #38bdf8;">📊 Statistiques Dataset</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
            <div style="background: #334155; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="font-size: 24px; font-weight: bold; color: #22c55e;">{stats.get('total_detected', 0)}</p>
                <p style="color: #94a3b8;">Détectés (depuis le démarrage)</p>
            </div>
            <div style="background: #334155; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="font-size: 24px; font-weight: bold; color: #eab308;">{stats.get('queue_size', stats.get('pending', 0))}</p>
                <p style="color: #94a3b8;">En attente</p>
            </div>
            <div style="background: #334155; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="font-size: 24px; font-weight: bold; color: #3b82f6;">{stats.get('validated', 0)}</p>
                <p style="color: #94a3b8;">Validés</p>
            </div>
        </div>
        <hr style="border-color: #475569; margin: 20px 0;">
        <h4 style="color: #94a3b8;">Dataset labellisé:</h4>
        <p>Total images: <strong>{dataset.get('total', 0)}</strong></p>
        <p>Taille: <strong>{dataset.get('size_mb', 0):.2f} MB</strong></p>
        {_retraining_line(stats.get("retraining"))}
    </div>
    """


def system_info(info: Dict[str, Any]) -> str:
    models = info.get("models", {})
    labels = {
        "plate_detector": "YOLO Plaque",
        "vehicle_detector": "YOLO Véhicule",
        "plate_reader": "OCR",
        "brand_classifier": "EfficientNet B0",
    }
    rows = "".join(
        f"<p><strong>{label}:</strong> {'✅' if models.get(role, {}).get('loaded') else '❌'} "
        f"<span style='color:#94a3b8;'>{models.get(role, {}).get('source', '')}</span></p>"
        for role, label in labels.items()
    )
    cells = "".join(
        f'<div style="background: #334155; padding: 15px; border-radius: 8px;">'
        f"<p><strong>{name}:</strong> {value}</p></div>"
        for name, value in (
            ("Environment", info.get("env")),
            ("Database", info.get("database")),
            ("Storage", info.get("storage")),
            ("Device", models.get("device")),
        )
    )
    return (f'<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">{cells}</div>'
            f'<hr style="margin: 15px 0; border-color: #475569;">{rows}')


# ═══════════════════════════════════════════════════════════════════════════════
# TABLEAUX
# ═══════════════════════════════════════════════════════════════════════════════

MISMATCH_COLUMNS = ["Plaque", "Prédite", "Déclarée", "Confiance", "Date"]
RESIDENT_COLUMNS = ["ID", "Plaque", "Marque", "Nom", "Prénom", "Accès"]
HISTORY_COLUMNS = ["Date", "Plaque", "Résultat", "Marque prédite", "Mismatch"]


def mismatch_table(records: Iterable[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {
            "Plaque": r.plaque,
            "Prédite": r.marque_predite,
            "Déclarée": r.marque_declaree,
            "Confiance": _percent(r.confiance),
            "Date": format_timestamp(r.timestamp),
        }
        for r in records
    ]
    return pd.DataFrame(rows, columns=MISMATCH_COLUMNS)


def residents_table(residents: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "ID": r.get("id", ""),
            "Plaque": r.get("plaque", ""),
            "Marque": r.get("marque_declaree", ""),
            "Nom": r.get("nom", ""),
            "Prénom": r.get("prenom", ""),
            "Accès": r.get("acces", ""),
        }
        for r in residents
    ]
    return pd.DataFrame(rows, columns=RESIDENT_COLUMNS)


def history_table(logs: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Date": format_timestamp(log.get("timestamp")),
            "Plaque": log.get("plaque", ""),
            "Résultat": log.get("resultat", ""),
            "Marque prédite": log.get("marque_predite", "") or "",
            "Mismatch": "⚠️" if log.get("mismatch") else "",
        }
        for log in logs
    ]
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)
