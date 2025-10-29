#!/usr/bin/env python3
"""
Team 2 — Plant Disease (Beans) (Image Classification)

Dataset: therealoise/bean-disease-dataset (Kaggle)
Modality: image (3-class)
Goal: Detect healthy vs two bean leaf diseases
Target: ≥92% test accuracy with a tiny CNN
"""

import os
import json
import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import os

# -------------------------
# 1️⃣ Load and Split Data
# -------------------------
def load_data(dataset_id="therealoise/bean-disease-dataset", val_split=0.15, test_split=0.15):
    """
    Download and prepare stratified train/val/test splits
    for a single-folder dataset with class subdirectories.
    """
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    from torchvision import datasets
    from torch.utils.data import Subset
    from collections import Counter
    import kagglehub

    path = kagglehub.dataset_download(dataset_id)
    print("✅ Dataset downloaded to:", path)

    # Single folder with class subfolders
    data_dir = os.path.join(path, "Bean_Dataset")
    full_dataset = datasets.ImageFolder(root=data_dir)
    labels = np.array([y for _, y in full_dataset.samples])

    # --- Stratified split: 70% train, 15% val, 15% test ---
    train_idx, temp_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_split + test_split,  # 30% held out
        stratify=labels,
        random_state=42,
    )
    val_rel = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_rel,  # split remaining 30% into 15%+15%
        stratify=labels[temp_idx],
        random_state=42,
    )

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    # --- Print class balance ---
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
    """Set up image transforms and dataloaders"""
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
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
# 3️⃣ Build Tiny CNN
# -------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# 4️⃣ Train
# -------------------------
def train(model, train_loader, val_loader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                correct += (outputs.argmax(1) == y).sum().item()

        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch}/10 | Train Loss: {train_loss/len(train_loader.dataset):.4f} | "
              f"Val Loss: {val_loss/len(val_loader.dataset):.4f} | Val Acc: {val_acc:.4f}")

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

    # Known class labels
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

    # --- Confusion Matrix ---
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

    return metrics


# -------------------------
# 6️⃣ Main Pipeline
# -------------------------
def main():
    train_ds, val_ds, test_ds = load_data()
    train_loader, val_loader, test_loader = preprocess(train_ds, val_ds, test_ds)
    model = TinyCNN(num_classes=3)
    model = train(model, train_loader, val_loader)
    metrics = evaluate(model, test_loader)

    print("\n📈 Final Metrics:")
    print(json.dumps(metrics["report"], indent=4))

    def main():
        train_ds, val_ds, test_ds = load_data()

        # 🔍 DEBUG: inspect split balance
        from collections import Counter
        def inspect_split(dataset, name):
            labels = [dataset.dataset.samples[i][1] for i in dataset.indices]
            print(f"{name} class distribution:", Counter(labels))

        inspect_split(train_ds, "Train")
        inspect_split(val_ds, "Val")
        inspect_split(test_ds, "Test")

        train_loader, val_loader, test_loader = preprocess(train_ds, val_ds, test_ds)
        model = TinyCNN(num_classes=3)
        model = train(model, train_loader, val_loader)
        metrics = evaluate(model, test_loader)

        print("\n📈 Final Metrics:")
        print(json.dumps(metrics["report"], indent=4))


if __name__ == "__main__":
    main()
