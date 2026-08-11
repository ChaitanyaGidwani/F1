"""Gradio Space - a standalone demo of the same pipeline the API serves.

Deployed to Hugging Face Spaces so judges have a link that works without
cloning the repo or running a server. It imports the *same* trend and
recommendation modules as the FastAPI backend, so there is one implementation
of the logic, not two that can drift.

Set MODEL_ID (or WW_MODEL_ID) to the fine-tuned Hub model id. If unset or
unreachable, it falls back to CLIP zero-shot and says so in the UI.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import gradio as gr
import pandas as pd
from PIL import Image

os.environ.setdefault("WW_MODEL_ID", os.environ.get("MODEL_ID", "models/vit-track-condition"))

from app.inference import load_classifier  # noqa: E402
from app.recommend import recommend  # noqa: E402
from app.trend import FramePrediction, TrackConditionTracker  # noqa: E402

CLASSIFIER = load_classifier()

BACKEND_NOTE = (
    "**Fine-tuned ViT** - trained on trackside imagery."
    if CLASSIFIER.backend == "fine-tuned"
    else "**CLIP zero-shot fallback** - the fine-tuned model was unavailable, "
    "so accuracy is materially lower (see the model card)."
)

TREND_TEXT = {
    "DRYING": "🟦 DRYING - conditions improving",
    "WETTING": "🟥 WETTING - conditions deteriorating",
    "STABLE": "⬜ STABLE",
    "INSUFFICIENT_DATA": "⏳ Collecting frames…",
}


def analyse(files: List[str], weather_hint: str):
    """Run an ordered sequence of frames through the tracker."""
    if not files:
        return None, "-", None, "Upload one or more track frames to begin.", None

    tracker = TrackConditionTracker()
    rows, gallery = [], []
    trend = None
    rec = None

    for index, path in enumerate(sorted(files)):
        image = Image.open(path).convert("RGB")
        probs = CLASSIFIER.predict(image)
        trend = tracker.update(FramePrediction.from_probs(probs))
        rec = recommend(trend, weather_hint=weather_hint or None)
        rows.append(
            {
                "frame": index + 1,
                "wetness": round(trend.wetness_history[-1], 3),
                "condition": trend.instant_class,
            }
        )
        gallery.append((image, f"{index + 1}. {trend.instant_class}"))

    # Report the smoothed condition rather than the last raw frame - the whole
    # point of the trend layer is that one frame is not the answer.
    label_out = {trend.current_class: float(trend.current_confidence)}

    summary = TREND_TEXT.get(trend.trend, trend.trend)
    if trend.transition:
        summary += f"  ·  {trend.transition.replace('->', ' → ')}"
    summary += f"  ·  {trend.frames} frames"

    message = f"### {rec.message}\n\n**Tyre:** {rec.tire}  **Action:** {rec.action}"

    return gallery, summary, pd.DataFrame(rows), message, label_out


with gr.Blocks(title="Weather Whiplash", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🏁 Weather Whiplash\n"
        "### Live track condition detector - Dry / Damp / Drying / Wet\n"
        f"{BACKEND_NOTE}\n\n"
        "Upload a **sequence** of trackside frames (numbered in time order). A single "
        "frame gives a condition; a sequence is what lets the trend layer decide "
        "whether the track is drying and when the tyre window opens."
    )

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Track frames (in time order)",
            )
            weather = gr.Textbox(
                label="Weather note (optional)",
                placeholder="e.g. radar shows rain in 10 minutes",
            )
            run = gr.Button("Analyse sequence", variant="primary")
        with gr.Column(scale=1):
            label = gr.Label(label="Current condition", num_top_classes=4)
            trend_md = gr.Markdown("-")
            rec_md = gr.Markdown("Upload one or more track frames to begin.")

    plot = gr.LinePlot(
        x="frame",
        y="wetness",
        title="Wetness over time (0 = Dry, 1 = Damp, 1.5 = Drying, 2 = Wet)",
        height=260,
        y_lim=[0, 2],
    )
    gallery = gr.Gallery(label="Frames", columns=6, height=190)

    run.click(
        analyse,
        inputs=[files, weather],
        outputs=[gallery, trend_md, plot, rec_md, label],
    )

if __name__ == "__main__":
    demo.launch()
