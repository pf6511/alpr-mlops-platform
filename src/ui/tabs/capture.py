"""Onglet « Image Processing » : traiter une photo et afficher le résultat."""

from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from src.services import Services
from src.ui import formatters as fmt

EMPTY = (None, None, None, None, "Please upload an image first.", "", "", "")


def build(services: Services) -> None:
    def on_process(image, conf_threshold, run_brand):
        if image is None:
            return EMPTY
        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        outcome = services.capture.process(image_bgr, conf_threshold=conf_threshold, run_brand=run_brand)

        raw, detection, plate_crop, final = fmt.pipeline_images(outcome.image_bgr, outcome.result, outcome.mismatch)
        return (
            raw, detection, plate_crop, final,
            fmt.ocr_summary(outcome.result),
            fmt.capture_report(outcome),
            fmt.vehicle_card(outcome.result.vehicle),
            fmt.brand_card(outcome.result.brand, outcome.mismatch, outcome.declared_brand),
        )

    with gr.Tab("📸 Image Processing"):
        with gr.Row():
            with gr.Column(scale=3, elem_classes="glass-card"):
                gr.Markdown("### 📤 Upload")
                image_input = gr.Image(type="pil", label="Upload Image")
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, step=0.05,
                    value=services.settings.models.detection_threshold,
                    label="Detection Threshold",
                )
                brand_checkbox = gr.Checkbox(value=True, label="🚗 Activer détection marque (Branche 2)")
                process_btn = gr.Button("🔍 Process Image", variant="primary")
                gr.Markdown("---")
                gr.Markdown("### 📊 OCR Result")
                ocr_output = gr.Textbox(label="Detected Plates", lines=3)

            with gr.Column(scale=7, elem_classes="glass-card"):
                gr.Markdown("### 🔬 Pipeline Steps")
                with gr.Tabs():
                    with gr.Tab("1️⃣ Raw Input"):
                        raw_output = gr.Image(label="Original Image")
                    with gr.Tab("2️⃣ YOLO Detection"):
                        gr.Markdown("*YOLOv8 identifies license plate regions*")
                        detection_output = gr.Image(label="Detection")
                    with gr.Tab("3️⃣ ROI Extraction"):
                        roi_output = gr.Image(label="Plate ROI")
                    with gr.Tab("4️⃣ Final + Brand"):
                        final_output = gr.Image(label="Final Result")
                        with gr.Row():
                            vehicle_output = gr.HTML(label="Véhicule")
                            brand_output = gr.HTML(label="Marque")

        with gr.Row():
            with gr.Column(elem_classes="glass-card"):
                gr.Markdown("### 📋 Analysis Report")
                report_output = gr.HTML()

        process_btn.click(
            fn=on_process,
            inputs=[image_input, conf_slider, brand_checkbox],
            outputs=[raw_output, detection_output, roi_output, final_output,
                     ocr_output, report_output, vehicle_output, brand_output],
        )
