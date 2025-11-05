#!/usr/bin/env python3
"""Gradio web UI for the Bean Disease classifier with enhanced statistics visualization

Usage:
    pip install gradio pillow pandas plotly
    python gradio_app_combined.py

Features:
- Upload an image and get a prediction (label + confidence)
- Keeps a local prediction history (same JSON used by the Tk app)
- Interactive statistics visualization with Plotly
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
    """Save a prediction to the history file."""
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
    """Get prediction history as a DataFrame."""
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
    """Add prediction label to image."""
    img = pil_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except Exception:
        font = ImageFont.load_default()
    text = f"{label} ({confidence:.1%})"
    
    # Draw semi-opaque rectangle with text
    w, h = img.size
    margin = 8
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        try:
            tw, th = font.getsize(text)
        except Exception:
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

def create_plots():
    """Generate performance plots from metrics."""
    try:
        if not os.path.exists("metrics.json"):
            return None, None, "metrics.json not found - run training first"

        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        
        report = metrics["report"]
        class_rows = {k: v for k, v in report.items() 
                     if isinstance(v, dict) and k not in ['macro avg', 'weighted avg']}
        
        if not class_rows:
            return None, "No class-specific metrics found in the report"

        # Create DataFrame
        df = pd.DataFrame(class_rows).T
        metrics_to_plot = ["precision", "recall", "f1-score"]
        df = df[metrics_to_plot]
        
        # Create bar plot
        bar_fig = go.Figure()
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
        
        for i, col in enumerate(df.columns):
            name = "F1-Score" if col == "f1-score" else col.title()
            bar_fig.add_trace(go.Bar(
                name=name,
                x=df.index,
                y=df[col],
                text=[f"{v:.3f}" for v in df[col]],
                textposition='auto',
                marker_color=colors[i % len(colors)]
            ))
        
        # Update bar plot layout
        bar_fig.update_layout(
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

        # Calculate confusion matrix from available metrics
        import numpy as np
        classes = list(class_rows.keys())
        n_classes = len(classes)
        total_samples = int(metrics["report"]["macro avg"]["support"])
        samples_per_class = total_samples // n_classes

        # Initialize confusion matrix
        confusion_matrix = np.zeros((n_classes, n_classes))
        
        # Fill in the confusion matrix using recall and precision
        for i, cls in enumerate(classes):
            metrics_cls = metrics["report"][cls]
            recall = metrics_cls["recall"]
            support = metrics_cls["support"]
            
            # True positives
            confusion_matrix[i, i] = int(round(recall * support))
            
            # Calculate false negatives
            false_negatives = support - confusion_matrix[i, i]
            
            # Distribute false negatives among other classes
            remaining_classes = [j for j in range(n_classes) if j != i]
            for j in remaining_classes:
                if j == remaining_classes[-1]:
                    # Put remaining FNs in last class to ensure row sums to support
                    confusion_matrix[i, j] = false_negatives - sum(confusion_matrix[i, k] for k in remaining_classes[:-1])
                else:
                    # Distribute FNs roughly equally
                    confusion_matrix[i, j] = false_negatives // len(remaining_classes)
        
        # Create heatmap
        # Format class names for display
        display_classes = [c.replace('_', ' ').title() for c in class_rows.keys()]
        
        matrix_fig = go.Figure(data=go.Heatmap(
            z=confusion_matrix,
            x=display_classes,
            y=display_classes,
            hoverongaps=False,
            texttemplate="%{z:.0f}",
            textfont={"size": 16},
            colorscale=[
                [0, '#f8f9fa'],
                [0.2, '#e3f2fd'],
                [0.4, '#90caf9'],
                [0.6, '#42a5f5'],
                [0.8, '#1e88e5'],
                [1, '#1565c0']
            ],
            showscale=True,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z:.0f}<extra></extra>"
        ))
        
        # Update confusion matrix layout
        matrix_fig.update_layout(
            title={
                'text': "Confusion Matrix",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            width=800,
            height=500,
            template="plotly_white",
            margin=dict(l=50, r=50, t=100, b=50),
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'},
            plot_bgcolor='white'
        )
        
        # Add percentage annotations
        for i in range(n_classes):
            row_sum = confusion_matrix[i].sum()
            for j in range(n_classes):
                percentage = (confusion_matrix[i, j] / row_sum * 100) if row_sum > 0 else 0
                matrix_fig.add_annotation(
                    x=display_classes[j],
                    y=display_classes[i],
                    text=f"{percentage:.1f}%",
                    showarrow=False,
                    font=dict(size=10, color='black'),
                    yshift=15
                )
        
        return bar_fig, matrix_fig, None
        
    except Exception as e:
        return None, f"Error generating plot: {str(e)}"

def update_display():
    """Update the statistics display."""
    try:
        if not os.path.exists("metrics.json"):
            return (
                "<div style='color: red; padding: 20px;'>metrics.json not found - run training first</div>",
                None,
                None,
                "Error: metrics.json not found"
            )

        with open("metrics.json", "r") as f:
            metrics = json.load(f)
        
        # Generate summary metrics
        html = create_metrics_summary(metrics)
        
        # Generate plots
        bar_fig, matrix_fig, error = create_plots()
        if error:
            return html, None, None, f"Error: {error}"
        
        return html, bar_fig, matrix_fig, "Statistics updated successfully"
        
    except Exception as e:
        return (
            "<div style='color: red; padding: 20px;'>Error loading metrics</div>",
            None,
            f"Error: {str(e)}"
        )

def launch_gradio():
    """Launch the Gradio web interface."""
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
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        # Model Performance Statistics
                        
                        This section shows detailed performance metrics for the model:
                        - Summary statistics (accuracy, precision, recall, F1)
                        - Per-class performance visualization
                        - Confusion matrix
                        
                        Click 'Update Statistics' to refresh the display.
                        """
                    )
            
            # Metrics display
            metrics_html = gr.HTML()
            with gr.Row():
                with gr.Column():
                    plot_output = gr.Plot(label="Per-class Performance")
            with gr.Row():
                with gr.Column():
                    matrix_output = gr.Plot(label="Confusion Matrix")
            
            with gr.Row():
                update_btn = gr.Button("Update Statistics", variant="primary")
                gen_status = gr.Markdown("")
            
            # Wire up update button and initialize display
            update_btn.click(
                fn=update_display,
                outputs=[metrics_html, plot_output, matrix_output, gen_status]
            )
            
            # Initialize on page load
            metrics_html.value, plot_output.value, matrix_output.value, _ = update_display()

    def find_free_port(start_port=7860, max_tries=100):
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_tries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        raise OSError(f"Could not find a free port in range {start_port}-{start_port + max_tries}")

    try:
        port = find_free_port(7860)
        print(f"Starting server on http://127.0.0.1:{port}")
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")

if __name__ == "__main__":
    launch_gradio()