"""Assemble les onglets en une application Gradio."""

from __future__ import annotations

import inspect

import gradio as gr

from src.services import Services

from . import theme
from .tabs import capture, history, residents, settings, validation


def _in_launch(param: str) -> bool:
    """Gradio 6 attend css/js dans launch() ; les versions antérieures, dans Blocks()."""
    return param in inspect.signature(gr.Blocks.launch).parameters


def _appearance() -> dict:
    return {"css": theme.CSS, "js": theme.FORCE_DARK_JS}


def build_app(services: Services) -> gr.Blocks:
    blocks_kwargs = {"title": theme.TITLE}
    blocks_kwargs.update({k: v for k, v in _appearance().items() if not _in_launch(k)})

    with gr.Blocks(**blocks_kwargs) as demo:
        with gr.Column(elem_classes="title-main"):
            gr.Markdown("# 🚗 Inside the ALPR Engine")
        with gr.Column(elem_classes="subtitle-main"):
            gr.Markdown("**LEAD Edition** - Détection Plaque + OCR + Marque + Mismatch Detection")

        with gr.Tabs():
            capture.build(services)
            validation.build(services, demo)
            residents.build(services, demo)
            settings.build(services)
            history.build(services, demo)

        with gr.Row():
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("""
                <div style="text-align: center; color: #64748b; font-size: 12px;">
                    <p>🚗 ALPR Engine - LEAD Edition |
                    YOLOv8 + fast-plate-ocr + EfficientNet B0</p>
                    <p>ACT-IA © 2025</p>
                </div>
                """)
    return demo


def launch(demo: gr.Blocks, services: Services) -> None:
    cfg = services.settings.gradio
    kwargs = dict(server_name=cfg.server_name, server_port=cfg.server_port, share=cfg.share, debug=cfg.debug)
    kwargs.update({k: v for k, v in _appearance().items() if _in_launch(k)})
    demo.launch(**kwargs)
