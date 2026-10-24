"""Onglet « Validation Marques » : traiter la file des incohérences."""

from __future__ import annotations

import gradio as gr

from src.services import Services
from src.services.validation import CUSTOM, REJECT, VALIDATE_DECLARED, VALIDATE_PREDICTED
from src.ui import formatters as fmt


def build(services: Services, demo: gr.Blocks) -> None:
    validation = services.validation

    def refresh_queue():
        return fmt.mismatch_table(validation.pending(limit=20))

    def refresh_stats():
        return fmt.dataset_stats(validation.stats())

    def act(action):
        def handler(plaque, correct_brand):
            # la décision met à jour la liste ET les statistiques
            return validation.act(plaque, action, correct_brand), refresh_queue(), refresh_stats()
        return handler

    def on_row_select(evt: gr.SelectData, table):
        if table is None or table.empty:
            return ""
        return str(table.iloc[evt.index[0]]["Plaque"])

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
                    btn_predicted = gr.Button("✅ Modèle correct", variant="primary")
                    btn_declared = gr.Button("📝 DB correcte", variant="secondary")
                    btn_custom = gr.Button("✏️ Marque saisie", variant="secondary")
                    btn_reject = gr.Button("❌ Rejeter", variant="stop")
                validation_msg = gr.Textbox(label="Résultat", interactive=False)

        with gr.Row():
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("#### 📋 Mismatches en attente")
                mismatch_table = gr.Dataframe(headers=fmt.MISMATCH_COLUMNS, interactive=False)
                refresh_queue_btn = gr.Button("🔄 Rafraîchir la liste")

        inputs = [selected_plaque, correct_brand]
        outputs = [validation_msg, mismatch_table, stats_html]
        btn_predicted.click(fn=act(VALIDATE_PREDICTED), inputs=inputs, outputs=outputs)
        btn_declared.click(fn=act(VALIDATE_DECLARED), inputs=inputs, outputs=outputs)
        btn_custom.click(fn=act(CUSTOM), inputs=inputs, outputs=outputs)
        btn_reject.click(fn=act(REJECT), inputs=inputs, outputs=outputs)

        mismatch_table.select(fn=on_row_select, inputs=[mismatch_table], outputs=[selected_plaque])
        refresh_stats_btn.click(fn=refresh_stats, outputs=[stats_html])
        refresh_queue_btn.click(fn=refresh_queue, outputs=[mismatch_table])
        demo.load(fn=refresh_stats, outputs=[stats_html])
        demo.load(fn=refresh_queue, outputs=[mismatch_table])
