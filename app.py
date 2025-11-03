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
from tkinter import filedialog, messagebox

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


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bean Leaf Health Checker")
        self.root.geometry("720x520")

        self.model: torch.nn.Module | None = None
        self.transform = get_transform()
        self.image_path: str | None = None
        self.tk_img: ImageTk.PhotoImage | None = None

        # UI Elements
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)

        self.select_btn = tk.Button(self.btn_frame, text="Alege imagine frunză...", command=self.on_select_image)
        self.select_btn.grid(row=0, column=1, padx=5)

        self.img_lbl = tk.Label(self.root)
        self.img_lbl.pack(pady=5)

        self.result_lbl = tk.Label(self.root, text="Rezultat: —", font=("Segoe UI", 12, "bold"))
        self.result_lbl.pack(pady=8)

        # Try auto-load weights once (bean_mobilenetv2* prioritizat)
        self.model, used_wp = load_model(None)
        if self.model is None:
            self.result_lbl.config(text="Model neîncărcat. Alege un fișier .pth cu greutăți.")
            messagebox.showwarning(
                "Greutăți lipsă",
                "Nu am găsit fișierul de greutăți 'bean_mobilenetv2(.pth|.pt)'.\n"
                "Rulează întâi antrenarea (python project.py) sau selectează manual fișierul .pth."
            )
        else:
            self.result_lbl.config(text=f"Model încărcat: {os.path.basename(used_wp)}. Alege o imagine pentru verificare.")

    def on_select_image(self):
        path = filedialog.askopenfilename(title="Alege imagine frunză",
                                          filetypes=[("Imagini", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"), ("Toate", "*.*")])
        if not path:
            return
        self.image_path = path
        try:
            img = Image.open(path).convert("RGB")
            disp = img.copy()
            disp.thumbnail((400, 400))
            self.tk_img = ImageTk.PhotoImage(disp)
            self.img_lbl.config(image=self.tk_img)
            self.result_lbl.config(text=f"Imagine selectată: {os.path.basename(path)}")
            # Auto-run prediction after selecting image
            self.on_predict()
        except Exception as e:
            messagebox.showerror("Eroare imagine", f"Nu pot citi imaginea.\n{e}")
            self.image_path = None

    def on_predict(self):
        if self.model is None:
            # Try once more to auto-load (bean_mobilenetv2 preferred)
            self.model, _ = load_model(None)
            if self.model is None:
                messagebox.showwarning(
                    "Model lipsă",
                    "Nu am găsit 'bean_mobilenetv2(.pth|.pt)'. Încarcă manual fișierul de greutăți din butonul 'Alege greutăți model...'."
                )
                return
        if not self.image_path:
            messagebox.showwarning("Imagine lipsă", "Te rog selectează o imagine de frunză.")
            return

        try:
            img = Image.open(self.image_path).convert("RGB")
            x = self.transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
        except Exception as e:
            messagebox.showerror("Predicție eșuată", f"A apărut o eroare la inferență.\n{e}")
            return

        pred_class = CLASS_NAMES[pred_idx]
        prob = float(probs[pred_idx])

        if pred_class == "healthy":
            msg = f"Rezultat: FĂRĂ PROBLEME (healthy) — confidență {prob:.1%}"
        else:
            msg = f"Probleme detectate: {pred_class} — confidență {prob:.1%}"
        self.result_lbl.config(text=msg)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
