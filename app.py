#!/usr/bin/env python3
"""
Bean Leaf Health Checker — Simple Desktop App (Tkinter)
- Select a leaf photo from your computer
- The app classifies it into one of: [angular_leaf_spot, bean_rust, healthy]
- It then reports if the leaf has problems (not healthy) or not

Requirements: torch, torchvision, pillow; Optional: a trained weights file `bean_mobilenetv2.pth` in project root
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk,filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
try:
    import cv2
except Exception:
    cv2 = None
import json
from datetime import datetime
import webbrowser

try:
    import plotly.graph_objects as go
    import pandas as pd
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
# Prefer importing the model builder from project.py to keep a single source of truth
try:
    from project import build_mobilenetv2  # type: ignore
except Exception as e:
    # Minimal fallback if import fails (should not happen in this repo)
    from torchvision import models
    def build_mobilenetv2(num_classes: int = 3, freeze_backbone: bool = True):
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        return model


CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
HISTORY_FILE="prediction_history.json"
DEFAULT_WEIGHT_CANDIDATES = [
    "bean_mobilenetv2.pth",
    "bean_mobilenetv2.pt",
    "bean_mobilenetv2",  # fără extensie (dacă a fost salvat așa)
    "model.pth",
    "mobilenetv2_bean.pth",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    # Match eval-time transforms with project.py
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def find_default_weights() -> str | None:
    for fname in DEFAULT_WEIGHT_CANDIDATES:
        if os.path.exists(fname):
            return fname
    return None


def load_model(weights_path: str | None = None) -> tuple[torch.nn.Module | None, str | None]:
    model = build_mobilenetv2(num_classes=len(CLASS_NAMES), freeze_backbone=True)
    model.to(device)

    # Try provided path or auto-detect
    wp = weights_path or find_default_weights()
    if wp is None:
        return None, None

    try:
        state = torch.load(wp, map_location=device)
        # allow both full model or state_dict
        if isinstance(state, dict) and all(k.startswith("features.") or k.startswith("classifier.") for k in state.keys()):
            model.load_state_dict(state)
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])  # common training wrapper
        else:
            # If someone saved the whole model
            if hasattr(state, "state_dict"):
                model = state
                model.to(device)
            else:
                # Last attempt: try as state_dict directly
                model.load_state_dict(state)
        model.eval()
        return model, wp
    except Exception as e:
        messagebox.showerror("Încărcare model eșuată", f"Nu pot încărca greutățile din '{wp}'.\n{e}")
        return None, None

class HistoryManager:
    def __init__(self, file_path=HISTORY_FILE):
        self.file_path = file_path
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def add_prediction(self, image_path, prediction, confidence, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'image_path': image_path,
            'prediction': prediction,
            'confidence': confidence
        }
        self.history.append(entry)
        self.save_history()
        return entry

    def save_history(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_history(self, limit=None):
        history = self.history[::-1]  # Most recent first
        return history[:limit] if limit else history

    def export_csv(self, filepath):
        if not self.history:
            return False
        try:
            df = pd.DataFrame(self.history)
            df.to_csv(filepath, index=False)
            return True
        except:
            return False
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Bean Leaf Disease Analyzer")
        self.root.geometry("1000x700")
        self.setup_style()
        
        self.model = None
        self.transform = get_transform()
        self.history_manager = HistoryManager()
        self.current_image = None
        self.current_image_path = None
        
        self.setup_ui()
        self.load_model()

    def setup_style(self):
        style = ttk.Style()
        style.configure('TNotebook', tabposition='n')
        style.configure('TButton', padding=6)
        style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Result.TLabel', font=('Segoe UI', 11))

    def setup_ui(self):
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Create tabs
        self.predict_tab = ttk.Frame(self.notebook)
        self.history_tab = ttk.Frame(self.notebook)
        self.stats_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.predict_tab, text='Predict')
        self.notebook.add(self.history_tab, text='History')
        self.notebook.add(self.stats_tab, text='Statistics')

        self.setup_predict_tab()
        self.setup_history_tab()
        self.setup_stats_tab()

    def setup_predict_tab(self):
        # Top frame for buttons
        btn_frame = ttk.Frame(self.predict_tab)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Select Image", command=self.on_select_image).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Open Webcam", command=self.on_webcam).pack(side='left', padx=5)

        # Image display frame
        self.display_frame = ttk.Frame(self.predict_tab)
        self.display_frame.pack(fill='both', expand=True, pady=10)

        # Left side: Original image
        self.orig_frame = ttk.LabelFrame(self.display_frame, text="Original Image")
        self.orig_frame.pack(side='left', fill='both', expand=True, padx=5)
        self.orig_label = ttk.Label(self.orig_frame)
        self.orig_label.pack(pady=5)

        # Right side: Processed image + controls
        self.proc_frame = ttk.LabelFrame(self.display_frame, text="Processed Image")
        self.proc_frame.pack(side='left', fill='both', expand=True, padx=5)
        self.proc_label = ttk.Label(self.proc_frame)
        self.proc_label.pack(pady=5)

        # Results frame
        results_frame = ttk.Frame(self.predict_tab)
        results_frame.pack(fill='x', pady=10, padx=10)
        
        self.result_label = ttk.Label(results_frame, text="Select an image to begin", style='Header.TLabel')
        self.result_label.pack()
        
        self.detail_label = ttk.Label(results_frame, text="", style='Result.TLabel')
        self.detail_label.pack()

    def setup_history_tab(self):
        # Controls frame
        controls = ttk.Frame(self.history_tab)
        controls.pack(fill='x', pady=5, padx=5)
        
        ttk.Button(controls, text="Export CSV", command=self.export_history).pack(side='left', padx=5)
        ttk.Button(controls, text="Refresh", command=self.refresh_history).pack(side='left', padx=5)

        # History view
        self.history_tree = ttk.Treeview(self.history_tab, columns=("Time", "Image", "Prediction", "Confidence"),
                                       show='headings')
        self.history_tree.heading("Time", text="Time")
        self.history_tree.heading("Image", text="Image")
        self.history_tree.heading("Prediction", text="Prediction")
        self.history_tree.heading("Confidence", text="Confidence")
        
        # Column widths
        self.history_tree.column("Time", width=150)
        self.history_tree.column("Image", width=200)
        self.history_tree.column("Prediction", width=150)
        self.history_tree.column("Confidence", width=100)
        
        self.history_tree.pack(fill='both', expand=True, padx=5, pady=5)
        self.refresh_history()

    def setup_stats_tab(self):
        if not PLOTLY_AVAILABLE:
            ttk.Label(self.stats_tab, text="Install plotly and pandas for interactive visualizations",
                     style='Header.TLabel').pack(pady=20)
            return

        # Buttons to show different visualizations
        btn_frame = ttk.Frame(self.stats_tab)
        btn_frame.pack(fill='x', pady=5, padx=5)
        
        ttk.Button(btn_frame, text="Show Confusion Matrix", 
                  command=lambda: self.show_plot("confusion_matrix.html")).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Show Metrics", 
                  command=lambda: self.show_plot("class_bars.html")).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Show Full Report", 
                  command=lambda: self.show_plot("report.html")).pack(side='left', padx=5)

        # Embed latest metrics if available
        if os.path.exists("metrics.json"):
            try:
                with open("metrics.json") as f:
                    metrics = json.load(f)
                self.show_metrics_summary(metrics)
            except:
                pass

    def show_metrics_summary(self, metrics):
        summary = f"""
        Overall Metrics:
        Accuracy: {metrics.get('accuracy', 0):.4f}
        Precision: {metrics.get('precision', 0):.4f}
        Recall: {metrics.get('recall', 0):.4f}
        F1 Score: {metrics.get('f1', 0):.4f}
        """
        ttk.Label(self.stats_tab, text=summary, style='Result.TLabel').pack(pady=20)

    def show_plot(self, filename):
        if os.path.exists(filename):
            webbrowser.open(filename)
        else:
            messagebox.showinfo("Info", f"Plot {filename} not found. Run training first.")

    def refresh_history(self):
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        for entry in self.history_manager.get_history():
            try:
                dt = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M')
            except:
                dt = entry['timestamp']
            self.history_tree.insert('', 'end', values=(
                dt,
                os.path.basename(entry['image_path']),
                entry['prediction'],
                f"{entry['confidence']:.1%}"
            ))

    def export_history(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            if self.history_manager.export_csv(filepath):
                messagebox.showinfo("Success", f"History exported to {filepath}")
            else:
                messagebox.showerror("Error", "Could not export history")

    def load_model(self):
        self.model, used_wp = load_model(None)
        if self.model is None:
            self.result_label.config(text="Model not loaded. Select weights file.")
            messagebox.showwarning(
                "Missing Weights",
                "Could not find model weights file.\nRun training first or select manually."
            )
        else:
            self.result_label.config(text=f"Model loaded: {os.path.basename(used_wp)}")

    def on_select_image(self):
        path = filedialog.askopenfilename(
            title="Select Leaf Image",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        self.current_image_path = path
        try:
            img = Image.open(path).convert("RGB")
            self.current_image = img
            self.update_image_display(img)
            self.result_label.config(text=f"Selected: {os.path.basename(path)}")
            self.predict_pil(img)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image.\n{e}")
            self.current_image_path = None

    def update_image_display(self, img, processed=None):
        # Original image
        display_size = (350, 350)
        orig_copy = img.copy()
        orig_copy.thumbnail(display_size)
        self.tk_orig = ImageTk.PhotoImage(orig_copy)
        self.orig_label.config(image=self.tk_orig)

        # Processed image (if provided, otherwise show same as original)
        if processed is None:
            processed = img
        proc_copy = processed.copy()
        proc_copy.thumbnail(display_size)
        self.tk_proc = ImageTk.PhotoImage(proc_copy)
        self.proc_label.config(image=self.tk_proc)

    def predict_pil(self, img: Image.Image):
        if self.model is None:
            self.model, _ = load_model(None)
            if self.model is None:
                messagebox.showwarning(
                    "Model Missing",
                    "Could not find model weights.\nRun training first or select manually."
                )
                return

        try:
            # Get the processed image for display
            processed = self.preprocess_for_display(img)
            self.update_image_display(img, processed)

            # Make prediction
            x = self.transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())

            pred_class = CLASS_NAMES[pred_idx]
            prob = float(probs[pred_idx])

            # Update UI with prediction
            if pred_class == "healthy":
                result = "HEALTHY LEAF"
                msg = f"No diseases detected (confidence: {prob:.1%})"
                color = "green"
            else:
                result = f"DISEASE DETECTED: {pred_class}"
                msg = f"Confidence: {prob:.1%}"
                color = "red"

            self.result_label.config(text=result)
            self.detail_label.config(text=msg)

            # Add to history
            self.history_manager.add_prediction(
                self.current_image_path,
                pred_class,
                prob
            )
            self.refresh_history()

        except Exception as e:
            messagebox.showerror("Prediction Failed", f"Error during prediction.\n{e}")

    def preprocess_for_display(self, img):
        """Show the preprocessing steps applied to the image"""
        display = img.copy()
        display = display.resize((224, 224))  # Match model input size
        return display

    def on_webcam(self):
        if cv2 is None:
            messagebox.showerror("OpenCV Missing", 
                               "opencv-python not installed.\nRun: pip install opencv-python")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam")
            return

        messagebox.showinfo("Webcam", "Press 'c' to capture, 'q' or ESC to quit")
        captured = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            disp = frame.copy()
            cv2.putText(disp, "Press 'c' to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow("Webcam", disp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                captured = frame.copy()
                break
            if key == ord('q') or key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured is None:
            return

        try:
            rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self.current_image = pil
            self.current_image_path = "webcam_capture.jpg"
            self.update_image_display(pil)
            self.predict_pil(pil)
        except Exception as e:
            messagebox.showerror("Error", f"Could not process webcam image.\n{e}")

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
