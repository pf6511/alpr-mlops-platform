"""
ALPR Engine - Validation UI Components
Composants Gradio réutilisables pour la validation des mismatches.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from configs.settings import get_settings

try:
    from src.models.mismatch_detector import get_mismatch_detector, MismatchRecord
    MISMATCH_AVAILABLE = True
except ImportError:
    MISMATCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def create_stats_card(title: str, value: str, color: str = "#3b82f6", icon: str = "📊") -> str:
    """Crée une carte de statistique HTML."""
    return f"""
    <div style="background: {color}22; border: 1px solid {color}; 
                border-radius: 12px; padding: 20px; text-align: center;">
        <p style="font-size: 14px; color: #94a3b8; margin: 0;">{icon} {title}</p>
        <p style="font-size: 32px; font-weight: bold; color: {color}; margin: 10px 0;">{value}</p>
    </div>
    """


def create_stats_grid(stats: Dict) -> str:
    """Crée une grille de statistiques HTML."""
    cards = [
        create_stats_card("Détectés", str(stats.get('total_detected', 0)), "#22c55e", "🔍"),
        create_stats_card("En attente", str(stats.get('pending', 0)), "#eab308", "⏳"),
        create_stats_card("Validés", str(stats.get('validated', 0)), "#3b82f6", "✅"),
        create_stats_card("Rejetés", str(stats.get('rejected', 0)), "#ef4444", "❌"),
    ]
    
    return f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
        {''.join(cards)}
    </div>
    """


def create_mismatch_card(record: 'MismatchRecord') -> str:
    """Crée une carte HTML pour un mismatch."""
    conf_percent = f"{record.confiance:.1%}" if record.confiance else "N/A"
    
    if record.confiance and record.confiance > 0.8:
        conf_color = "#22c55e"
    elif record.confiance and record.confiance > 0.5:
        conf_color = "#eab308"
    else:
        conf_color = "#ef4444"
    
    return f"""
    <div style="background: #1e293b; border: 1px solid #475569; border-radius: 12px; 
                padding: 20px; margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; color: #f1f5f9;">🚗 {record.plaque}</h3>
            <span style="background: {conf_color}33; color: {conf_color}; 
                         padding: 5px 10px; border-radius: 20px; font-size: 12px;">
                {conf_percent}
            </span>
        </div>
        <hr style="border-color: #475569; margin: 15px 0;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <p style="color: #94a3b8; margin: 5px 0; font-size: 12px;">Prédite:</p>
                <p style="color: #f97316; font-size: 18px; font-weight: bold; margin: 0;">
                    {record.marque_predite or 'N/A'}
                </p>
            </div>
            <div>
                <p style="color: #94a3b8; margin: 5px 0; font-size: 12px;">Déclarée:</p>
                <p style="color: #3b82f6; font-size: 18px; font-weight: bold; margin: 0;">
                    {record.marque_declaree or 'N/A'}
                </p>
            </div>
        </div>
        <p style="color: #64748b; font-size: 12px; margin-top: 15px;">
            📅 {record.timestamp[:19] if record.timestamp else 'N/A'}
        </p>
    </div>
    """


def create_dataset_progress(stats: Dict) -> str:
    """Crée une barre de progression vers le retraining."""
    settings = get_settings()
    min_size = settings.airflow.min_dataset_size
    
    dataset_stats = stats.get('dataset', {})
    total = dataset_stats.get('total', 0)
    
    progress = min(100, (total / min_size) * 100) if min_size > 0 else 0
    color = "#22c55e" if progress >= 100 else "#eab308"
    ready = stats.get('ready_for_retraining', False)
    
    return f"""
    <div style="background: #1e293b; border-radius: 12px; padding: 20px; margin: 15px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <h4 style="margin: 0; color: #f1f5f9;">📈 Progression retraining</h4>
            <span style="color: {color};">{total} / {min_size}</span>
        </div>
        <div style="background: #334155; border-radius: 10px; height: 20px; overflow: hidden;">
            <div style="background: {color}; width: {progress}%; height: 100%;"></div>
        </div>
        <p style="color: #94a3b8; font-size: 12px; margin-top: 10px; text-align: center;">
            {'✅ Prêt!' if ready else f'⏳ {max(0, min_size - total)} restantes'}
        </p>
    </div>
    """


def create_brand_distribution(stats: Dict) -> str:
    """Crée un affichage de la distribution par marque."""
    by_label = stats.get('dataset', {}).get('by_label', {})
    
    if not by_label:
        return "<p style='color: #64748b; text-align: center;'>Aucune donnée</p>"
    
    sorted_labels = sorted(by_label.items(), key=lambda x: x[1], reverse=True)
    max_count = max(by_label.values()) if by_label else 1
    
    bars_html = ""
    for label, count in sorted_labels[:10]:
        width = (count / max_count) * 100
        bars_html += f"""
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <span style="width: 100px; color: #f1f5f9; font-size: 12px;">{label.capitalize()}</span>
            <div style="flex: 1; background: #334155; border-radius: 4px; height: 20px; margin: 0 10px;">
                <div style="background: #3b82f6; width: {width}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <span style="color: #94a3b8; font-size: 12px; width: 40px; text-align: right;">{count}</span>
        </div>
        """
    
    return f"""
    <div style="background: #1e293b; border-radius: 12px; padding: 20px; margin: 15px 0;">
        <h4 style="margin: 0 0 15px 0; color: #f1f5f9;">📊 Distribution par marque</h4>
        {bars_html}
    </div>
    """


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def refresh_validation_data() -> Tuple[str, str, str, str]:
    """Rafraîchit toutes les données de validation."""
    if not MISMATCH_AVAILABLE:
        empty = "<p style='color: #64748b;'>Mismatch detector non disponible</p>"
        return empty, empty, empty, empty
    
    detector = get_mismatch_detector()
    stats = detector.get_stats()
    
    stats_html = create_stats_grid(stats)
    progress_html = create_dataset_progress(stats)
    distribution_html = create_brand_distribution(stats)
    
    pending = detector.get_pending(limit=10)
    if pending:
        mismatch_html = "".join([create_mismatch_card(r) for r in pending])
    else:
        mismatch_html = """
        <div style="text-align: center; padding: 40px; color: #64748b;">
            <p style="font-size: 48px;">✅</p>
            <p>Aucun mismatch en attente</p>
        </div>
        """
    
    return stats_html, progress_html, distribution_html, mismatch_html


def handle_validation(plaque: str, action: str, custom_brand: str = "") -> Tuple[str, str, str, str, str]:
    """Gère une action de validation."""
    if not MISMATCH_AVAILABLE:
        return "Non disponible", "", "", "", ""
    
    if not plaque:
        return "⚠️ Sélectionner une plaque", *refresh_validation_data()
    
    detector = get_mismatch_detector()
    pending = detector.get_pending(limit=100)
    
    record = None
    for r in pending:
        if r.plaque == plaque:
            record = r
            break
    
    if not record:
        return f"❌ {plaque} non trouvée", *refresh_validation_data()
    
    if action == 'model':
        success, msg = detector.validate_as_predicted(record)
    elif action == 'db':
        success, msg = detector.validate_as_declared(record)
    elif action == 'custom' and custom_brand:
        success, msg = detector.validate(record, custom_brand)
    elif action == 'reject':
        success, msg = detector.reject(record, "Manual rejection")
    else:
        msg = "⚠️ Action invalide"
    
    return msg, *refresh_validation_data()


def get_mismatch_dataframe(limit: int = 50) -> 'pd.DataFrame':
    """Récupère les mismatches sous forme de DataFrame."""
    if not MISMATCH_AVAILABLE or not PANDAS_AVAILABLE:
        return pd.DataFrame()
    
    detector = get_mismatch_detector()
    pending = detector.get_pending(limit=limit)
    
    data = []
    for r in pending:
        data.append({
            'Plaque': r.plaque,
            'Prédite': r.marque_predite,
            'Déclarée': r.marque_declaree,
            'Confiance': f"{r.confiance:.1%}" if r.confiance else "N/A",
            'Date': r.timestamp[:19] if r.timestamp else ''
        })
    
    return pd.DataFrame(data)


def get_dataset_dataframe() -> 'pd.DataFrame':
    """Récupère les stats du dataset sous forme de DataFrame."""
    if not MISMATCH_AVAILABLE or not PANDAS_AVAILABLE:
        return pd.DataFrame()
    
    detector = get_mismatch_detector()
    stats = detector.get_dataset_stats()
    
    by_label = stats.get('by_label', {})
    data = [{'Marque': k.capitalize(), 'Images': v} for k, v in by_label.items()]
    
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('Images', ascending=False)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

def create_quick_validation_widget():
    """Crée un widget de validation rapide."""
    if not GRADIO_AVAILABLE:
        raise RuntimeError("Gradio non disponible")
    
    with gr.Accordion("🔄 Validation rapide", open=False) as accordion:
        queue_size = gr.Number(label="En attente", interactive=False)
        
        with gr.Row():
            plaque_input = gr.Textbox(label="Plaque", scale=2)
            action_dropdown = gr.Dropdown(
                choices=["✅ Modèle correct", "📝 DB correcte", "❌ Rejeter"],
                label="Action", scale=1
            )
        
        validate_btn = gr.Button("Valider", variant="primary", size="sm")
        result_text = gr.Textbox(label="Résultat", interactive=False)
    
    def update_queue_size():
        if MISMATCH_AVAILABLE:
            return get_mismatch_detector().get_queue_size()
        return 0
    
    def quick_validate(plaque, action):
        action_map = {
            "✅ Modèle correct": "model",
            "📝 DB correcte": "db",
            "❌ Rejeter": "reject"
        }
        action_key = action_map.get(action, "reject")
        msg, *_ = handle_validation(plaque, action_key)
        return msg, update_queue_size()
    
    return {
        'accordion': accordion,
        'queue_size': queue_size,
        'plaque_input': plaque_input,
        'action_dropdown': action_dropdown,
        'validate_btn': validate_btn,
        'result_text': result_text,
        'update_queue_size': update_queue_size,
        'quick_validate': quick_validate
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Test Validation UI Components...")
    
    test_stats = {
        'total_detected': 42,
        'pending': 5,
        'validated': 30,
        'rejected': 7,
        'dataset': {
            'total': 250,
            'by_label': {'renault': 50, 'peugeot': 45, 'citroen': 40},
            'size_mb': 125.5
        },
        'ready_for_retraining': False
    }
    
    print("Stats grid:", create_stats_grid(test_stats)[:50] + "...")
    print("Progress:", create_dataset_progress(test_stats)[:50] + "...")
    
    print("✅ Tests OK")
