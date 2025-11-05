#!/usr/bin/env python3
"""Gradio web UI for the Bean Disease classifier

Usage:
    pip install gradio pillow pandas plotly
    python gradio_app.py

Features:
- Upload an image and get a prediction (label + confidence)
- Keeps a local prediction history (same JSON used by the Tk app)
- Shows interactive HTML report (report.html) if present
"""
import os
import json
import time
from datetime import datetime
from io import BytesIO

import gradio as gr
import socket
from PIL import Image, ImageDraw, ImageFont
import torch
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Configure plotly template
pio.templates.default = "plotly_white"

# Reuse helpers/constants from the desktop app where possible
try:
    from app import get_transform, load_model, CLASS_NAMES, HISTORY_FILE
except Exception:
    # If app.py is not importable, fallback to local definitions
    from project import build_mobilenetv2 as _build

    def get_transform():
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def load_model(wp=None):
        # Try to load the saved weights bean_mobilenetv2.pth
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = _build(num_classes=3, freeze_backbone=True)
            model.to(device)
            wp = wp or "bean_mobilenetv2.pth"
            if os.path.exists(wp):
                state = torch.load(wp, map_location=device)
                if isinstance(state, dict) and "state_dict" in state:
                    model.load_state_dict(state["state_dict"])
                elif isinstance(state, dict):
                    model.load_state_dict(state)
                else:
                    model = state
                model.eval()
                return model, wp
        except Exception:
            pass
        return None, None

    CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
    HISTORY_FILE = "prediction_history.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = get_transform()

# Load model once at startup (if available)
MODEL, MODEL_WEIGHTS = load_model(None)


def save_history_entry(entry: dict):
    data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_history_df():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=["timestamp", "image", "prediction", "confidence"])
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Most recent first
        data = list(reversed(data))
        df = pd.DataFrame(data)
        if "image_path" in df.columns:
            df = df.rename(columns={"image_path": "image"})
        df["confidence"] = df["confidence"].map(lambda v: f"{v:.2%}")
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "image", "prediction", "confidence"])


def annotate_image(pil_img: Image.Image, label: str, confidence: float) -> Image.Image:
    img = pil_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except Exception:
        font = ImageFont.load_default()
    text = f"{label} ({confidence:.1%})"
    # draw semi-opaque rectangle
    w, h = img.size
    margin = 8
    # textsize() may not exist in all Pillow versions; prefer textbbox or font.getsize
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        try:
            tw, th = font.getsize(text)
        except Exception:
            # fallback estimate
            tw, th = (len(text) * 7, 12)
    rect_h = th + margin * 2
    rect_w = tw + margin * 2
    rect_x = 10
    rect_y = h - rect_h - 10
    draw.rectangle([rect_x, rect_y, rect_x + rect_w, rect_y + rect_h], fill=(0, 0, 0, 140))
    draw.text((rect_x + margin, rect_y + margin), text, fill=(255, 255, 255, 255), font=font)
    return img.convert("RGB")


def predict(image: Image.Image):
    """Run model prediction on a PIL image and return annotated image, label, confidence."""
    global MODEL
    if image is None:
        return None, "No image", "0%"

    # If model not loaded, try loading again
    if MODEL is None:
        MODEL, _ = load_model(None)
        if MODEL is None:
            return None, "Model not loaded", "0%"

    # Run transform and inference
    try:
        x = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = MODEL(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        label = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx])

        # Save uploaded image to uploads/ for traceability
        os.makedirs("uploads", exist_ok=True)
        ts = int(time.time())
        fname = f"uploads/upload_{ts}.jpg"
        try:
            image.save(fname)
            save_name = fname
        except Exception:
            save_name = "(in-memory)"

        # Save history entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "image_path": save_name,
            "prediction": label,
            "confidence": conf,
        }
        try:
            save_history_entry(entry)
        except Exception:
            pass

        annotated = annotate_image(image, label, conf)
        return annotated, label, f"{conf:.1%}"
    except Exception as e:
        return None, f"Error: {e}", "0%"


def read_html_report(path: str):
    """Read an HTML report and return an embeddable fragment.
    
    If the file is a full HTML document, extract the contents of the <body> so
    the fragment can be embedded inside Gradio's HTML component. Otherwise
    return the file contents unchanged.
    """
    if not os.path.exists(path):
        return f"Report file not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # If it's a full HTML document, extract the body inner HTML to embed
        lower = content.lower()
        if "<body" in lower and "</body>" in lower:
            # find start of <body...>
            start = lower.find("<body")
            start = content.find('>', start) + 1
            end = lower.rfind("</body>")
            if start > 0 and end > start:
                return content[start:end]

        return content
    except Exception as e:
        return f"Could not read report: {e}"


def create_plot():
    """Generate a bar plot from metrics."""
    try:
        if not os.path.exists("metrics.json"):
            return None, "metrics.json not found - run training first"

        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        
        report = metrics.get("report", {})
        class_rows = {k: v for k, v in report.items() 
                     if isinstance(v, dict) and k not in ['macro avg', 'weighted avg']}
        
        if not class_rows:
            return None, "No class-specific metrics found in the report"

        # Create DataFrame
        df = pd.DataFrame(class_rows).T
        metrics_to_plot = ["precision", "recall", "f1-score"]
        df = df[metrics_to_plot]
        
        # Create plot
        fig = go.Figure()
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
        
        for i, col in enumerate(df.columns):
            name = "F1-Score" if col == "f1-score" else col.title()
            fig.add_trace(go.Bar(
                name=name,
                x=df.index,
                y=df[col],
                text=[f"{v:.3f}" for v in df[col]],
                textposition='auto',
                marker_color=colors[i % len(colors)]
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Per-class Performance Metrics",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title="Class",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            barmode='group',
            width=800,
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        return fig, None
        
    except Exception as e:
        return None, f"Error generating plot: {str(e)}"

def create_metrics_summary(metrics):
    """Create HTML for metrics summary."""
    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px;">
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">Accuracy</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('accuracy', 0):.4f}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">Precision</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('precision', 0):.4f}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">Recall</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('recall', 0):.4f}</div>
        </div>
        <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="color: #666; font-size: 0.9em;">F1 Score</div>
            <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('f1', 0):.4f}</div>
        </div>
    </div>
    """

def generate_reports_from_metrics():
    """Generate interactive report with metrics summary and plots."""
    try:
        if not os.path.exists("metrics.json"):
            return "", "metrics.json not found - run training first"

        with open("metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
        # Create metrics summary
        content = create_metrics_summary(metrics)
        
        # Create plot
        fig, error = create_plot()
        if error:
            return content + f"<div style='color: red;'>{error}</div>", "Failed to generate plot"
            
        # Convert plot to HTML
        plot_html = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'responsive': True}
        )

    # If confusion matrix png exists, embed it; otherwise note absence
    cm_html = ""
    if os.path.exists("confusion_matrix.png"):
        try:
            # Verify image can be opened
            Image.open("confusion_matrix.png")
            # Use data URL to embed image directly in HTML
            with open("confusion_matrix.png", "rb") as img_file:
                import base64
                img_data = base64.b64encode(img_file.read()).decode()
                cm_html = f'''
                    <div style="margin: 20px 0;">
                        <h3>Confusion Matrix</h3>
                        <img src="data:image/png;base64,{img_data}" 
                             alt="confusion matrix" 
                             style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px;"/>
                    </div>
                '''
        except Exception as e:
            cm_html = f"<p>Error loading confusion_matrix.png: {e}</p>"
    else:
        cm_html = "<p>No confusion_matrix.png found.</p>"

    summary_html = f"""
    <div style="font-family: Arial, Helvetica, sans-serif; margin-bottom: 24px; background-color: #f8f9fa; padding: 20px; border-radius: 8px;">
        <h2 style="color: #2c3e50; margin-bottom: 16px;">Bean Disease Classification Report</h2>
        <p style="color: #666; margin-bottom: 16px;">Generated from metrics.json</p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
            <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="color: #666; font-size: 0.9em;">Accuracy</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('accuracy', 0):.4f}</div>
            </div>
            <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="color: #666; font-size: 0.9em;">Precision</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('precision', 0):.4f}</div>
            </div>
            <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="color: #666; font-size: 0.9em;">Recall</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('recall', 0):.4f}</div>
            </div>
            <div style="background: white; padding: 16px; border-radius: 6px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="color: #666; font-size: 0.9em;">F1 Score</div>
                <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{metrics.get('f1', 0):.4f}</div>
            </div>
        </div>
    </div>
    """

    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>Classification Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 0;
                padding: 24px;
                background-color: #ffffff;
                color: #2c3e50;
                line-height: 1.6;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .section {{
                margin-bottom: 32px;
                background: white;
                padding: 24px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 0;
            }}
            hr {{
                border: 0;
                height: 1px;
                background-color: #eee;
                margin: 32px 0;
            }}
            .plotly-graph-div {{
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {summary_html}
            <div class="section">
                {cm_html}
            </div>
            <div class="section">
                <h3>Per-class Performance Metrics</h3>
                {plot_html}
            </div>
        </div>
    </body>
    </html>
    """

    try:
        with open("report.html", "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        # If writing the full report failed, still try to return the embeddable fragment
        frag = summary_html + cm_html + plot_html
        return (frag, f"Failed to write report.html: {e}")

    # Return an embeddable fragment for Gradio plus status message
    # Add a container div to ensure proper plot sizing and scrolling
    fragment = f"""
    <div style="max-width: 100%; overflow-x: auto;">
        {summary_html}
        {cm_html}
        <div style="margin: 20px 0;">
            {plot_html}
        </div>
    </div>
    """
    return (fragment, "report.html generated")


def launch_gradio():
    with gr.Blocks(title="Bean Leaf Disease — Gradio UI") as demo:
        gr.Markdown("# Bean Leaf Disease — Demo UI")

        with gr.Tab("Predict"):
            inp = gr.Image(type="pil", label="Upload leaf image")
            out_img = gr.Image(type="pil", label="Annotated prediction")
            lbl = gr.Textbox(label="Predicted class")
            conf = gr.Textbox(label="Confidence")
            btn = gr.Button("Predict")
            btn.click(predict, inputs=inp, outputs=[out_img, lbl, conf])

        with gr.Tab("History"):
            hist_df = gr.Dataframe(value=get_history_df(), label="Prediction history")
            refresh = gr.Button("Refresh")
            refresh.click(lambda: get_history_df(), inputs=None, outputs=hist_df)
            with gr.Row():
                export_btn = gr.Button("Export CSV")
                export_msg = gr.Textbox(label="Export status")

            def export_history():
                df = get_history_df()
                if df.empty:
                    return "No history to export"
                out_path = f"history_export_{int(time.time())}.csv"
                try:
                    df.to_csv(out_path, index=False)
                    return f"Saved to {out_path}"
                except Exception as e:
                    return f"Export failed: {e}"

            export_btn.click(export_history, inputs=None, outputs=export_msg)

        with gr.Tab("Statistics"):
            gr.Markdown("### Interactive training report (if available)")
            report_html = gr.HTML(value=read_html_report("report.html"))
            gen_btn = gr.Button("(Re)generate reports from metrics.json")
            gen_status = gr.Textbox(label="Status")
            
            # Add Plotly script tag at the top level
            plotly_script = gr.HTML("""
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            """)
            
            gr.Markdown("---")
            gr.Markdown("Below: the current report (live-updated after generation)")
            # wire button to generator
            gen_btn.click(fn=generate_reports_from_metrics, inputs=None, outputs=[report_html, gen_status])

    # Try a different port since 7860 might be in use
    port = 7861
    
    print(f"Starting Gradio on http://127.0.0.1:{port}")
    demo.launch(
        server_name="127.0.0.1", 
        server_port=port, 
        share=False,
        quiet=True,
        show_error=True
    )


if __name__ == "__main__":
    launch_gradio()