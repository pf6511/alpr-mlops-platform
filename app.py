"""
Inside the ALPR Engine - Interactive Demo (LEAD Edition)
Pipeline complet: Détection plaque + OCR + Véhicule + Marque + Mismatch Detection

Structure: src/, configs/, pipelines/
"""

import gradio as gr
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - Nouvelle structure
# ═══════════════════════════════════════════════════════════════════════════════

# Ajouter le projet au path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
from configs.settings import get_settings

# Data layer
from src.data.database import DatabaseManager

# Models
from src.models.pipeline import ALPRPipeline

# Utils
from src.utils.visualizer import (
    create_step_images, 
    create_analysis_report,
    format_ocr_result
)
from src.utils.error_gallery import ErrorGallery
from src.utils.video_processor import (
    create_annotated_video,
    process_gif,
    sample_video_frames,
    create_static_video
)
from src.utils.access_control import AccessController

# Optionnels
try:
    from src.models.mismatch_detector import get_mismatch_detector
    MISMATCH_AVAILABLE = True
except ImportError:
    MISMATCH_AVAILABLE = False

try:
    from src.data.storage import get_storage
    STORAGE_AVAILABLE = True
except ImportError:
    STORAGE_AVAILABLE = False

import tempfile
import textwrap
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Outputs directory
try:
    os.makedirs("outputs", exist_ok=True)
    print("✅ Outputs directory OK")
except Exception as e:
    print(f"❌ Error: {e}")

print("🚀 Launching ALPR Engine (LEAD Edition)...")

# Initialize components
print("📦 Initializing Pipeline (with Branch 2)...")
pipeline = ALPRPipeline(load_branch2=True)

print("📦 Initializing Error Gallery...")
error_gallery = ErrorGallery()

print("📦 Initializing Access Controller...")
access_controller = AccessController()

print("📦 Initializing Database...")
db = DatabaseManager()

if MISMATCH_AVAILABLE:
    print("📦 Initializing Mismatch Detector...")
    mismatch_detector = get_mismatch_detector()
else:
    mismatch_detector = None

settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def format_brand_result(brand_result: dict, mismatch: bool, declared_brand: str) -> str:
    """Formate le résultat de classification marque en HTML."""
    if not brand_result:
        return "<p style='color: #888;'>🚗 Marque: Non détectée</p>"
    
    brand = brand_result.get('brand', 'Inconnu')
    confidence = brand_result.get('confidence', 0)
    top3 = brand_result.get('top3', [])
    
    # Couleur selon mismatch
    if mismatch:
        color = "#ef4444"  # Rouge
        icon = "⚠️"
        status = f"MISMATCH (déclarée: {declared_brand})"
    else:
        color = "#22c55e"  # Vert
        icon = "✅"
        status = "OK"
    
    html = f"""
    <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                border: 2px solid {color}; border-radius: 12px; padding: 15px; margin: 10px 0;">
        <h3 style="margin: 0; color: {color};">{icon} Classification Marque: {status}</h3>
        <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">{brand}</p>
        <p style="margin: 5px 0;">Confiance: <strong>{confidence:.1%}</strong></p>
        <hr style="border-color: {color}33;">
        <p style="font-size: 12px; color: #888;">Top 3:</p>
        <ul style="margin: 5px 0; padding-left: 20px;">
    """
    
    for item in top3[:3]:
        b = item[0] if isinstance(item, tuple) else item.get('brand', '?')
        c = item[1] if isinstance(item, tuple) else item.get('confidence', 0)
        html += f"<li>{b}: {c:.1%}</li>"
    
    html += "</ul></div>"
    
    return html


def format_vehicle_result(vehicle_detection: dict) -> str:
    """Formate le résultat de détection véhicule."""
    if not vehicle_detection:
        return "<p style='color: #888;'>🚙 Véhicule: Non détecté</p>"
    
    class_name = vehicle_detection.get('class_name', 'vehicle')
    confidence = vehicle_detection.get('confidence', 0)
    
    return f"""
    <div style="background: #3b82f622; border: 1px solid #3b82f6; 
                border-radius: 8px; padding: 10px; margin: 5px 0;">
        <p style="margin: 0;"><strong>🚙 Véhicule détecté:</strong> {class_name.capitalize()}</p>
        <p style="margin: 5px 0; font-size: 12px;">Confiance: {confidence:.1%}</p>
    </div>
    """


def process_upload(image, conf_threshold=0.5, run_branch2=True):
    """
    Process uploaded image through ALPR pipeline.
    
    Returns:
        Tuple of outputs for Gradio interface
    """
    if image is None:
        return None, None, None, None, "Please upload an image first.", None, "", ""
    
    # Save temporary image
    temp_path = "/tmp/upload_temp.jpg"
    image.save(temp_path)
    
    # Get declared brand from DB if plate is known
    declared_brand = None
    
    # Process through pipeline
    results = pipeline.process_image(temp_path, conf_threshold, run_branch2=run_branch2)
    
    # Extract images for each step
    step1_img = results['step1_raw']
    step2_img = results['step2_detection']
    step5_img = results['step5_final']
    step3_img = results['step3_roi'][0] if results['step3_roi'] else None
    
    # Get plate text for DB lookup
    plate_text = ""
    if results['step4_ocr']:
        plate_text = results['step4_ocr'][0]['text']
        
        # Lookup declared brand in DB
        resident = db.get_plate_with_brand(plate_text)
        if resident:
            declared_brand = resident.get('marque_declaree')
    
    # Re-process with declared brand for mismatch detection
    if declared_brand and run_branch2:
        results = pipeline.process_image(temp_path, conf_threshold, 
                                         run_branch2=True, declared_brand=declared_brand)
        step5_img = results['step5_final']
    
    # Create OCR results text
    ocr_text = ""
    if results['step4_ocr']:
        for i, ocr_result in enumerate(results['step4_ocr'], 1):
            ocr_text += format_ocr_result(
                ocr_result['text'], 
                ocr_result['confidence']
            )
            ocr_text += "\n\n"
    else:
        ocr_text = "❌ No plates detected"
    
    # Create analysis report
    report = create_analysis_report(results)
    
    # Access Control Check
    if results and results['step4_ocr']:
        best_plate = results['step4_ocr'][0]['text']
        authorized, message = access_controller.check_access(best_plate)
        
        status_color = "green" if authorized else "red"
        status_icon = "✅" if authorized else "⛔"
        
        # Log the attempt
        access_controller.log_attempt(best_plate, authorized)
        
        # Log with brand info
        if run_branch2 and results.get('brand_result'):
            brand_result = results['brand_result']
            db.add_log(
                plaque=best_plate,
                authorized=authorized,
                normalized=best_plate.replace("-", "").replace(" ", ""),
                marque_predite=brand_result.get('brand'),
                marque_confiance=brand_result.get('confidence'),
                mismatch=results.get('mismatch', False)
            )
        
        access_banner = f"""<div class="access-banner" style="background-color: {status_color}; 
            color: white; text-align: center; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
            {status_icon} {message}
        </div>"""
        report = access_banner + "\n\n" + report
    else:
        report = """<div class="access-banner" style="background-color: #64748b; 
            color: white; text-align: center; padding: 10px; border-radius: 8px;">
            ⚠️ NO PLATE DETECTED - ACCESS UNKNOWN
        </div>""" + "\n\n" + report
    
    # Branch 2 results
    vehicle_html = format_vehicle_result(results.get('vehicle_detection'))
    brand_html = format_brand_result(
        results.get('brand_result'),
        results.get('mismatch', False),
        declared_brand or "Non déclarée"
    )
    
    # Mismatch handling
    if results.get('mismatch') and mismatch_detector:
        vehicle_crop = results.get('vehicle_crop')
        mismatch_detector.detect_and_record(
            plaque=plate_text,
            marque_predite=results['brand_result']['brand'],
            marque_declaree=declared_brand,
            confiance=results['brand_result']['confidence'],
            vehicle_crop=vehicle_crop
        )
        
        # Add mismatch warning to report
        mismatch_warning = """
        <div style="background: #fef2f2; border: 2px solid #ef4444; 
                    border-radius: 8px; padding: 15px; margin: 10px 0;">
            <h3 style="color: #dc2626; margin: 0;">⚠️ MISMATCH DÉTECTÉ</h3>
            <p>La marque prédite ne correspond pas à la marque déclarée.</p>
            <p>Ce cas a été ajouté à la file de validation.</p>
        </div>
        """
        report = mismatch_warning + report
    
    # Handle Error Gallery
    gallery_update = gr.update()
    if not results['step4_ocr']:
        try:
            error_gallery.add_example(
                image_path=temp_path,
                issue="No Plate Detected",
                expected="Unknown",
                predicted="None",
                detection_conf=0.0,
                ocr_conf=0.0,
                analysis=f"Failed to detect plate."
            )
            gallery_update = error_gallery.create_gallery_markdown()
        except Exception as e:
            print(f"Error updating gallery: {e}")
    
    return step1_img, step2_img, step3_img, step5_img, ocr_text, report, vehicle_html, brand_html


def get_mismatch_queue():
    """Récupère la file d'attente des mismatches pour validation."""
    if not mismatch_detector:
        return pd.DataFrame(columns=['Plaque', 'Prédite', 'Déclarée', 'Confiance', 'Date'])
    
    pending = mismatch_detector.get_pending(limit=20)
    
    data = []
    for record in pending:
        data.append({
            'Plaque': record.plaque,
            'Prédite': record.marque_predite,
            'Déclarée': record.marque_declaree,
            'Confiance': f"{record.confiance:.1%}",
            'Date': record.timestamp[:19] if record.timestamp else ''
        })
    
    return pd.DataFrame(data)


def validate_mismatch(plaque: str, correct_brand: str, action: str):
    """Valide ou rejette un mismatch."""
    if not mismatch_detector:
        return "Mismatch detector non disponible", get_mismatch_queue()
    
    pending = mismatch_detector.get_pending(limit=100)
    
    for record in pending:
        if record.plaque == plaque:
            if action == "validate_predicted":
                success, msg = mismatch_detector.validate_as_predicted(record)
            elif action == "validate_declared":
                success, msg = mismatch_detector.validate_as_declared(record)
            elif action == "reject":
                success, msg = mismatch_detector.reject(record, "Manual rejection")
            else:
                success, msg = mismatch_detector.validate(record, correct_brand)
            
            return msg, get_mismatch_queue()
    
    return f"Plaque {plaque} non trouvée", get_mismatch_queue()


def get_dataset_stats():
    """Récupère les stats du dataset labellisé."""
    if not mismatch_detector:
        return "Dataset stats non disponibles"
    
    stats = mismatch_detector.get_stats()
    
    html = f"""
    <div style="background: #1e293b; padding: 20px; border-radius: 12px;">
        <h3 style="color: #38bdf8;">📊 Statistiques Dataset</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 15px;">
            <div style="background: #334155; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="font-size: 24px; font-weight: bold; color: #22c55e;">{stats.get('total_detected', 0)}</p>
                <p style="color: #94a3b8;">Détectés</p>
            </div>
            <div style="background: #334155; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="font-size: 24px; font-weight: bold; color: #eab308;">{stats.get('pending', 0)}</p>
                <p style="color: #94a3b8;">En attente</p>
            </div>
            <div style="background: #334155; padding: 15px; border-radius: 8px; text-align: center;">
                <p style="font-size: 24px; font-weight: bold; color: #3b82f6;">{stats.get('validated', 0)}</p>
                <p style="color: #94a3b8;">Validés</p>
            </div>
        </div>
        <hr style="border-color: #475569; margin: 20px 0;">
        <h4 style="color: #94a3b8;">Dataset labellisé:</h4>
        <p>Total images: <strong>{stats.get('dataset', {}).get('total', 0)}</strong></p>
        <p>Taille: <strong>{stats.get('dataset', {}).get('size_mb', 0):.2f} MB</strong></p>
        <p>Prêt pour retraining: <strong>{'✅ Oui' if stats.get('ready_for_retraining') else '❌ Non'}</strong></p>
    </div>
    """
    
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.glass-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 20px;
}

.title-main {
    text-align: center;
    padding: 20px;
}

.subtitle-main {
    text-align: center;
    color: #94a3b8;
}

.access-banner {
    padding: 15px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
}

.brand-card {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid #3b82f6;
    border-radius: 12px;
    padding: 15px;
}

.mismatch-warning {
    background: rgba(239, 68, 68, 0.1);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 15px;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(css=custom_css, title="ALPR Engine - LEAD Edition") as demo:
    
    # Header
    with gr.Column(elem_classes="title-main"):
        gr.Markdown("# 🚗 Inside the ALPR Engine")
    with gr.Column(elem_classes="subtitle-main"):
        gr.Markdown("**LEAD Edition** - Détection Plaque + OCR + Marque + Mismatch Detection")
    
    # Main Tabs
    with gr.Tabs():
        
        # ═══════════════════════════════════════════════════════════════════════
        # TAB 1: Image Processing
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("📸 Image Processing"):
            with gr.Row():
                # Left panel - Controls
                with gr.Column(scale=3, elem_classes="glass-card"):
                    gr.Markdown("### 📤 Upload")
                    image_input = gr.Image(type="pil", label="Upload Image")
                    
                    conf_slider = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Detection Threshold"
                    )
                    
                    branch2_checkbox = gr.Checkbox(
                        value=True, 
                        label="🚗 Activer détection marque (Branche 2)"
                    )
                    
                    process_btn = gr.Button("🔍 Process Image", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 📊 OCR Result")
                    ocr_output = gr.Textbox(label="Detected Plates", lines=3)
                
                # Right panel - Results
                with gr.Column(scale=7, elem_classes="glass-card"):
                    gr.Markdown("### 🔬 Pipeline Steps")
                    
                    with gr.Tabs():
                        with gr.Tab("1️⃣ Raw Input"):
                            step1_output = gr.Image(label="Original Image")
                        
                        with gr.Tab("2️⃣ YOLO Detection"):
                            gr.Markdown("*YOLOv8 identifies license plate regions*")
                            step2_output = gr.Image(label="Detection")
                        
                        with gr.Tab("3️⃣ ROI Extraction"):
                            step3_output = gr.Image(label="Plate ROI")
                        
                        with gr.Tab("4️⃣ Final + Brand"):
                            step4_output = gr.Image(label="Final Result")
                            
                            with gr.Row():
                                vehicle_output = gr.HTML(label="Véhicule")
                                brand_output = gr.HTML(label="Marque")
            
            with gr.Row():
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### 📋 Analysis Report")
                    report_output = gr.HTML()
            
            # Process button click
            process_btn.click(
                fn=process_upload,
                inputs=[image_input, conf_slider, branch2_checkbox],
                outputs=[
                    step1_output, step2_output, step3_output, step4_output,
                    ocr_output, report_output, vehicle_output, brand_output
                ]
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # TAB 2: Validation Marques
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("🚗 Validation Marques"):
            gr.Markdown("### 🔄 File de validation des mismatches")
            gr.Markdown("*Valider/rejeter les erreurs de classification pour améliorer le modèle*")
            
            with gr.Row():
                with gr.Column(scale=2, elem_classes="glass-card"):
                    gr.Markdown("#### 📊 Statistiques")
                    stats_html = gr.HTML()
                    refresh_stats_btn = gr.Button("🔄 Rafraîchir", variant="secondary")
                
                with gr.Column(scale=3, elem_classes="glass-card"):
                    gr.Markdown("#### ⚡ Actions rapides")
                    
                    selected_plaque = gr.Textbox(label="Plaque sélectionnée")
                    correct_brand = gr.Textbox(label="Marque correcte (si autre)")
                    
                    with gr.Row():
                        btn_validate_pred = gr.Button("✅ Modèle correct", variant="primary")
                        btn_validate_decl = gr.Button("📝 DB correcte", variant="secondary")
                        btn_reject = gr.Button("❌ Rejeter", variant="stop")
                    
                    validation_msg = gr.Textbox(label="Résultat", interactive=False)
            
            with gr.Row():
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("#### 📋 Mismatches en attente")
                    mismatch_table = gr.Dataframe(
                        headers=['Plaque', 'Prédite', 'Déclarée', 'Confiance', 'Date'],
                        interactive=False
                    )
                    refresh_queue_btn = gr.Button("🔄 Rafraîchir la liste")
            
            # Event handlers
            refresh_stats_btn.click(fn=get_dataset_stats, outputs=[stats_html])
            refresh_queue_btn.click(fn=get_mismatch_queue, outputs=[mismatch_table])
            
            btn_validate_pred.click(
                fn=lambda p: validate_mismatch(p, "", "validate_predicted"),
                inputs=[selected_plaque],
                outputs=[validation_msg, mismatch_table]
            )
            btn_validate_decl.click(
                fn=lambda p: validate_mismatch(p, "", "validate_declared"),
                inputs=[selected_plaque],
                outputs=[validation_msg, mismatch_table]
            )
            btn_reject.click(
                fn=lambda p: validate_mismatch(p, "", "reject"),
                inputs=[selected_plaque],
                outputs=[validation_msg, mismatch_table]
            )
            
            # Load initial data
            demo.load(fn=get_dataset_stats, outputs=[stats_html])
            demo.load(fn=get_mismatch_queue, outputs=[mismatch_table])
        
        # ═══════════════════════════════════════════════════════════════════════
        # TAB 3: Administration Clients
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("👥 Administration Clients"):
            with gr.Row():
                gr.Markdown("### 👥 Gestion des résidents et véhicules")
            
            with gr.Row():
                with gr.Column(scale=3, elem_classes="glass-card"):
                    gr.Markdown("#### ➕ Ajouter un résident")
                    
                    add_plaque = gr.Textbox(label="Plaque", placeholder="AA-123-BB")
                    add_marque = gr.Textbox(label="Marque véhicule", placeholder="Renault")
                    add_nom = gr.Textbox(label="Nom")
                    add_prenom = gr.Textbox(label="Prénom")
                    add_tel = gr.Textbox(label="Téléphone")
                    add_adresse = gr.Textbox(label="Adresse")
                    add_ville = gr.Textbox(label="Ville")
                    add_cp = gr.Textbox(label="Code postal")
                    add_age = gr.Number(label="Âge", value=30)
                    add_abo = gr.Checkbox(label="Abonné")
                    add_acces = gr.Checkbox(label="Accès autorisé", value=True)
                    
                    add_btn = gr.Button("➕ Ajouter", variant="primary")
                    add_result = gr.Textbox(label="Résultat", interactive=False)
                
                with gr.Column(scale=5, elem_classes="glass-card"):
                    gr.Markdown("#### 📋 Liste des résidents")
                    residents_table = gr.Dataframe(
                        headers=['ID', 'Plaque', 'Marque', 'Nom', 'Prénom', 'Accès'],
                        interactive=False
                    )
                    refresh_residents_btn = gr.Button("🔄 Rafraîchir")
            
            def add_resident(plaque, marque, nom, prenom, tel, adresse, ville, cp, age, abo, acces):
                try:
                    db.add_resident({
                        'plaque': plaque,
                        'nom': nom,
                        'prenom': prenom,
                        'age': int(age),
                        'telephone': tel,
                        'adresse': adresse,
                        'ville': ville,
                        'code_postal': cp,
                        'abonnement': "oui" if abo else "non",
                        'acces': "oui" if acces else "non",
                        'marque_declaree': marque
                    })
                    return f"✅ Résident {prenom} {nom} ajouté!", get_residents_df()
                except Exception as e:
                    return f"❌ Erreur: {e}", get_residents_df()
            
            def get_residents_df():
                try:
                    residents = db.get_all_residents() 
                    data = []
                    for r in residents[:50]:
                        data.append({
                            'ID': r.get('id', ''),
                            'Plaque': r.get('plaque', ''),
                            'Marque': r.get('marque_declaree', ''),
                            'Nom': r.get('nom', ''),
                            'Prénom': r.get('prenom', ''),
                            'Accès': r.get('acces', '')
                        })
                    return pd.DataFrame(data)
                except Exception as e:
                    print(f"Error get_residents_df: {e}")
                    return pd.DataFrame()
            
            add_btn.click(
                fn=add_resident,
                inputs=[add_plaque, add_marque, add_nom, add_prenom, add_tel,
                        add_adresse, add_ville, add_cp, add_age, add_abo, add_acces],
                outputs=[add_result, residents_table]
            )
            refresh_residents_btn.click(fn=get_residents_df, outputs=[residents_table])
            demo.load(fn=get_residents_df, outputs=[residents_table])
        
        # ═══════════════════════════════════════════════════════════════════════
        # TAB 4: Settings
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("⚙️ Settings"):
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("### 🔧 Configuration des modèles")
                
                available_models = pipeline.get_available_models()
                current_model = os.path.basename(pipeline.model_path) if pipeline.model_path else "N/A"
                
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=current_model if current_model in available_models else (available_models[0] if available_models else None),
                    label="Modèle YOLO Plaque"
                )
                
                reload_model_btn = gr.Button("🔄 Recharger le modèle")
                model_status = gr.Textbox(value=f"Modèle actuel: {current_model}", label="Statut")
                
                def reload_model_handler(model_name):
                    success, msg = pipeline.reload_model(model_name)
                    return msg
                
                reload_model_btn.click(
                    fn=reload_model_handler,
                    inputs=[model_dropdown],
                    outputs=[model_status]
                )
            
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("### 📊 Informations système")
                
                info_html = f"""
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                    <div style="background: #334155; padding: 15px; border-radius: 8px;">
                        <p><strong>Environment:</strong> {settings.env}</p>
                    </div>
                    <div style="background: #334155; padding: 15px; border-radius: 8px;">
                        <p><strong>Database:</strong> {settings.database.mode}</p>
                    </div>
                    <div style="background: #334155; padding: 15px; border-radius: 8px;">
                        <p><strong>Storage:</strong> {settings.s3.mode}</p>
                    </div>
                    <div style="background: #334155; padding: 15px; border-radius: 8px;">
                        <p><strong>Device:</strong> {pipeline.device}</p>
                    </div>
                </div>
                <hr style="margin: 15px 0; border-color: #475569;">
                <p><strong>YOLO Plaque:</strong> {'✅' if pipeline.yolo_plate else '❌'}</p>
                <p><strong>YOLO Véhicule:</strong> {'✅' if pipeline.yolo_vehicle else '❌'}</p>
                <p><strong>OCR:</strong> {'✅' if pipeline.ocr_model else '❌'}</p>
                <p><strong>EfficientNet:</strong> {'✅' if pipeline.brand_model else '❌'}</p>
                """
                
                gr.HTML(info_html)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TAB 5: History
        # ═══════════════════════════════════════════════════════════════════════
        with gr.Tab("📜 History"):
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("### 📜 Historique des accès")
                
                def get_history_df():
                    try:
                        logs = db.get_recent_logs(limit=50)
                        data = []
                        for log in logs:
                            data.append({
                                'Date': log.get('timestamp', '')[:19],
                                'Plaque': log.get('plaque', ''),
                                'Résultat': log.get('resultat', ''),
                                'Marque prédite': log.get('marque_predite', ''),
                                'Mismatch': '⚠️' if log.get('mismatch') else ''
                            })
                        return pd.DataFrame(data)
                    except Exception as e:
                        return pd.DataFrame()
                
                history_table = gr.Dataframe(
                    headers=['Date', 'Plaque', 'Résultat', 'Marque prédite', 'Mismatch'],
                    interactive=False
                )
                refresh_history_btn = gr.Button("🔄 Rafraîchir")
                
                refresh_history_btn.click(fn=get_history_df, outputs=[history_table])
                demo.load(fn=get_history_df, outputs=[history_table])
    
    # Footer
    with gr.Row():
        with gr.Column(elem_classes="glass-card"):
            gr.Markdown("""
            <div style="text-align: center; color: #64748b; font-size: 12px;">
                <p>🚗 ALPR Engine - LEAD Edition | 
                YOLOv8 + fast-plate-ocr + EfficientNet B4</p>
                <p>ACT-IA © 2025</p>
            </div>
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    gradio_config = settings.gradio
    
    demo.launch(
        server_name=gradio_config.server_name,
        server_port=gradio_config.server_port,
        share=gradio_config.share,
        debug=gradio_config.debug
    )
