"""Onglet « Settings » : choix du modèle de plaques et informations système."""

from __future__ import annotations

import gradio as gr

from src.services import Services
from src.ui import formatters as fmt


def build(services: Services) -> None:
    system = services.system

    def on_reload(model_name):
        _, message = system.reload_plate_model(model_name)
        return message, fmt.system_info(system.info())

    with gr.Tab("⚙️ Settings"):
        with gr.Column(elem_classes="glass-card"):
            gr.Markdown("### 🔧 Configuration des modèles")
            choices = system.available_plate_models()
            current = system.current_plate_model()
            model_dropdown = gr.Dropdown(
                choices=choices,
                value=current if current in choices else (choices[0] if choices else None),
                label="Modèle YOLO Plaque",
            )
            reload_btn = gr.Button("🔄 Recharger le modèle")
            model_status = gr.Textbox(value=f"Modèle actuel: {current}", label="Statut")

        with gr.Column(elem_classes="glass-card"):
            gr.Markdown("### 📊 Informations système")
            info_html = gr.HTML(fmt.system_info(system.info()))

        reload_btn.click(fn=on_reload, inputs=[model_dropdown], outputs=[model_status, info_html])
