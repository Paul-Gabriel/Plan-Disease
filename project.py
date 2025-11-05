#!/usr/bin/env python3
"""
Team 2 — Bean Plant Disease Classification 🌱
Dataset: therealoise/bean-disease-dataset (Kaggle)
Goal: Classify bean leaves as healthy or diseased (3 classes)
Model: Enhanced TinyCNN (VGG-style)
"""

# -------------------------
# Imports
# -------------------------
import os
import json
import numpy as np
from collections import Counter
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import pandas as pd
import plotly.graph_objects as go
import datetime


# -------------------------
# 1️⃣ Load and Split Data
# -------------------------
def load_data(dataset_id="therealoise/bean-disease-dataset", val_split=0.1, test_split=0.1):
    """Download and prepare stratified train/val/test splits."""
    path = kagglehub.dataset_download(dataset_id)
    print("✅ Dataset downloaded to:", path)

    data_dir = os.path.join(path, "Bean_Dataset")
    full_dataset = datasets.ImageFolder(root=data_dir)
    labels = np.array([y for _, y in full_dataset.samples])

    train_idx, temp_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_split + test_split,
        stratify=labels,
        random_state=42,
    )
    val_rel = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_rel,
        stratify=labels[temp_idx],
        random_state=42,
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    # Print class balance
    def show_class_balance(dataset, name):
        lbls = [dataset.dataset.samples[i][1] for i in dataset.indices]
        counts = Counter(lbls)
        print(f"  {name} class balance:", counts)

    total = len(full_dataset)
    print(f"📊 Stratified split (total={total}):")
    print(f"  Train={len(train_ds)} ({len(train_ds)/total:.1%})")
    print(f"  Val={len(val_ds)} ({len(val_ds)/total:.1%})")
    print(f"  Test={len(test_ds)} ({len(test_ds)/total:.1%})")

    show_class_balance(train_ds, "Train")
    show_class_balance(val_ds, "Val")
    show_class_balance(test_ds, "Test")

    return train_ds, val_ds, test_ds


# -------------------------
# 2️⃣ Preprocessing / Augmentation
# -------------------------
def preprocess(train_ds, val_ds, test_ds, batch_size=32):
    """Set up transforms and dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(40),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_ds.dataset.transform = transform_train
    val_ds.dataset.transform = transform_test
    test_ds.dataset.transform = transform_test

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# -------------------------
# 3️⃣ Model: MobileNetV2 Transfer Learning
# -------------------------
def build_mobilenetv2(num_classes=3, freeze_backbone=True):
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# -------------------------
# 4️⃣ Train Model (unchanged)
# -------------------------
def train(model, train_loader, val_loader, epochs=25, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        batches = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", unit="batch", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            batches+=X.size(0)

            tqdm.write("") if False else None

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", unit="batch", leave=False):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()
        tqdm.write(
            f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
    return model


# -------------------------
# 5️⃣ Evaluate
# -------------------------
def evaluate(model, test_loader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    all_labels = list(range(len(class_names)))

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("✅ Confusion matrix saved as confusion_matrix.png")

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("✅ Metrics saved to metrics.json")
    print_metrics_rich(metrics)

    return (metrics,cm)


# -------------------------
# 6️⃣ Main
# -------------------------
def main():
    train_ds, val_ds, test_ds = load_data()
    train_loader, val_loader, test_loader = preprocess(train_ds, val_ds, test_ds)
    model = build_mobilenetv2(num_classes=3, freeze_backbone=False)
    model = train(model, train_loader, val_loader, epochs=15, lr=1e-4)
    # Save trained weights for the GUI app (app.py)
    try:
        torch.save(model.state_dict(), "bean_mobilenetv2.pth")
        print("✅ Model weights saved to bean_mobilenetv2.pth")
    except Exception as e:
        print("⚠️ Could not save model weights:", e)
    metrics,cm = evaluate(model, test_loader)
    class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
    generate_html_report(metrics, cm.tolist() if hasattr(cm, "tolist") else cm, class_names)

    print("\n📈 Final Metrics:")
    print(json.dumps(metrics["report"], indent=4))
    try:
        out = plot_metrics_bars(metrics, out_path="class_bars.html")
        if out:
            print(f"✅ Per-class metric bars saved to {out}")
            # webbrowser.open_new_tab(out)   # optional
    except Exception as e:
        print("⚠️  Plotly bars failed:", e)

# -------------------------
# 7.  Results Visualization & Reporting
# -------------------------
def print_metrics_rich(metrics: dict):
    """
    Nicely print classification report + summary using rich.
    Expects `metrics` to contain keys: 'accuracy','precision','recall','f1','report'
    where 'report' is the sklearn classification_report as dict.
    """
    console = Console()
    # Summary panel
    summary = (
        f"[bold green]Accuracy:[/bold green] {metrics.get('accuracy', 0):.4f}\n"
        f"[bold yellow]Precision (weighted):[/bold yellow] {metrics.get('precision', 0):.4f}\n"
        f"[bold cyan]Recall (weighted):[/bold cyan] {metrics.get('recall', 0):.4f}\n"
        f"[bold magenta]F1 (weighted):[/bold magenta] {metrics.get('f1', 0):.4f}\n"
    )
    console.print(Panel(summary, title="Final metrics", expand=False))

    # Classification report table
    report = metrics.get("report", {})
    # If report contains top-level keys such as 'accuracy' or 'macro avg', separate class rows vs summary rows
    class_rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    if class_rows:
        table = Table(title="Classification Report", show_lines=True)
        table.add_column("Class", style="bold")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1-score", justify="right")
        table.add_column("Support", justify="right")
        for cls, row in class_rows.items():
            p = row.get("precision", 0)
            r = row.get("recall", 0)
            f = row.get("f1-score", 0)
            s = int(row.get("support", 0))
            table.add_row(cls, f"{p:.3f}", f"{r:.3f}", f"{f:.3f}", str(s))
        console.print(table)

def plot_metrics_bars(metrics: dict, out_path: str = "class_bars.html"):
    report = metrics.get("report", {})
    class_rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    if not class_rows:
        return None
    df = pd.DataFrame(class_rows).T[["precision", "recall", "f1-score"]].rename(columns={"f1-score": "f1"})
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Bar(name=col, x=df.index.tolist(), y=df[col].tolist()))
    fig.update_layout(barmode="group", title="Per-class Precision / Recall / F1", yaxis_title="Score", width=800, height=500)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path

# -------------------------
# 8. HTML Report Generation
# -------------------------
def generate_html_report(metrics: dict, cm: list, class_names: list, out_path: str = "report.html"):
    """
    Create a single self-contained HTML file with:
     - interactive Plotly confusion matrix
     - HTML table for classification report (pandas-styled)
     - small header with timestamp and summary
    Returns the written path.
    """
    # Prepare classification table
    report = metrics.get("report", {})
    # Convert class rows to DataFrame (skip macro/weighted/accuracy entries if present)
    class_rows = {k: v for k, v in report.items() if isinstance(v, dict)}
    if class_rows:
        df = pd.DataFrame(class_rows).T  # columns: precision, recall, f1-score, support
        # nicer column names
        df = df.rename(columns={"f1-score": "f1", "support": "support"})
        df_html = df.style.format({
            "precision": "{:.3f}",
            "recall": "{:.3f}",
            "f1": "{:.3f}",
            "support": "{:.0f}"
        }).set_caption("Per-class metrics").to_html()
    else:
        df_html = "<p>No class-level report available</p>"

    # Plotly confusion matrix
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            reversescale=False,
            hovertemplate="Pred: %{x}<br>True: %{y}<br>Count: %{z}<extra></extra>"
        )
    )
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True", width=700, height=600)

    # Convert Plotly fig to html fragment (no full_html so we can build our own page)
    plotly_fragment = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # Small summary block
    summary_html = f"""
    <div style="font-family: Arial, sans-serif; margin-bottom: 12px;">
      <h2>Bean Disease Classification Report</h2>
      <p>Generated: {datetime.datetime.now().isoformat()}</p>
      <p><strong>Accuracy:</strong> {metrics.get('accuracy', 0):.4f} &nbsp;
         <strong>Precision:</strong> {metrics.get('precision', 0):.4f} &nbsp;
         <strong>Recall:</strong> {metrics.get('recall', 0):.4f} &nbsp;
         <strong>F1:</strong> {metrics.get('f1', 0):.4f}</p>
    </div>
    """

    # Combine into a simple HTML page
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Classification Report</title>
      <style>
        body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; }}
        .container {{ display: flex; gap: 24px; align-items: flex-start; }}
        .left {{ flex: 1 1 420px; }}
        .right {{ flex: 1 1 320px; }}
      </style>
    </head>
    <body>
      {summary_html}
      <div class="container">
        <div class="left">
          {plotly_fragment}
        </div>
        <div class="right">
          {df_html}
        </div>
      </div>
    </body>
    </html>
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


if __name__ == "__main__":
    main()
