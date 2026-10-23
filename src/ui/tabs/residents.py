"""Onglet « Administration Clients » : ajouter et lister les résidents."""

from __future__ import annotations

import gradio as gr

from src.services import Services
from src.ui import formatters as fmt


def build(services: Services, demo: gr.Blocks) -> None:
    residents = services.residents

    def refresh():
        return fmt.residents_table(residents.list(limit=50))

    def on_add(plaque, marque, nom, prenom, tel, adresse, ville, cp, age, abonne, acces):
        _, message = residents.add({
            "plaque": plaque,
            "nom": nom,
            "prenom": prenom,
            "age": int(age or 0),
            "telephone": tel,
            "adresse": adresse,
            "ville": ville,
            "code_postal": cp,
            "abonnement": "oui" if abonne else "non",
            "acces": "oui" if acces else "non",
            "marque_declaree": marque,
        })
        return message, refresh()

    with gr.Tab("👥 Administration Clients"):
        gr.Markdown("### 👥 Gestion des résidents et véhicules")

        with gr.Row():
            with gr.Column(scale=3, elem_classes="glass-card"):
                gr.Markdown("#### ➕ Ajouter un résident")
                fields = [
                    gr.Textbox(label="Plaque", placeholder="AA-123-BB"),
                    gr.Textbox(label="Marque véhicule", placeholder="Renault"),
                    gr.Textbox(label="Nom"),
                    gr.Textbox(label="Prénom"),
                    gr.Textbox(label="Téléphone"),
                    gr.Textbox(label="Adresse"),
                    gr.Textbox(label="Ville"),
                    gr.Textbox(label="Code postal"),
                    gr.Number(label="Âge", value=30),
                    gr.Checkbox(label="Abonné"),
                    gr.Checkbox(label="Accès autorisé", value=True),
                ]
                add_btn = gr.Button("➕ Ajouter", variant="primary")
                add_result = gr.Textbox(label="Résultat", interactive=False)

            with gr.Column(scale=5, elem_classes="glass-card"):
                gr.Markdown("#### 📋 Liste des résidents")
                table = gr.Dataframe(headers=fmt.RESIDENT_COLUMNS, interactive=False)
                refresh_btn = gr.Button("🔄 Rafraîchir")

        add_btn.click(fn=on_add, inputs=fields, outputs=[add_result, table])
        refresh_btn.click(fn=refresh, outputs=[table])
        demo.load(fn=refresh, outputs=[table])
