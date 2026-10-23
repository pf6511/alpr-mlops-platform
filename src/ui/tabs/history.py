"""Onglet « History » : derniers passages enregistrés."""

from __future__ import annotations

import gradio as gr

from src.services import Services
from src.ui import formatters as fmt


def build(services: Services, demo: gr.Blocks) -> None:
    def refresh():
        return fmt.history_table(services.history.recent(limit=50))

    with gr.Tab("📜 History"):
        with gr.Column(elem_classes="glass-card"):
            gr.Markdown("### 📜 Historique des accès")
            table = gr.Dataframe(headers=fmt.HISTORY_COLUMNS, interactive=False)
            refresh_btn = gr.Button("🔄 Rafraîchir")

        refresh_btn.click(fn=refresh, outputs=[table])
        demo.load(fn=refresh, outputs=[table])
