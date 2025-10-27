#!/usr/bin/env python3
"""
Team 2 — Plant Disease (Beans) (Image Classification)

Dataset: nateraw/beans   (Hugging Face)
Modality: image (3-class)

Goal
----
Detect healthy vs two bean leaf diseases. Target: ≥92% test accuracy with a tiny CNN.

Suggested path
--------------
- Small CNN (2–3 conv layers + ReLU + max-pool + dropout).
- Light augmentation (flip, rotate ≤ 15°, small color jitter).
- Confusion matrix + per-class precision/recall.

Stretch ideas
-------------
- Swap in a small pretrained backbone and compare.

Team roles (recommended)
------------------------
A) Data collection & prep  -> B) Training & visualization  -> C) Testing & validation
Tip: do a 1-hour micro-rotation mid-week so everyone touches each stage.

Deliverables
------------
1) Single script: project.py (this file is your starting point).
2) Short README (1 page) with results table + 1–2 plots.
3) Saved artifacts (model + vectorizer/preprocessor), and a note on dataset license/ethics.

Notes
-----
- Use a fixed random seed for reproducibility.
"""

# -------------------------
# Minimal function skeletons (fill these)
# -------------------------

def load_data(dataset_id: str = "nateraw/beans"):
    """Load the dataset from Hugging Face and return raw splits.
    TODO: implement using datasets.load_dataset(dataset_id)
    Return the structure you prefer (e.g., DatasetDict or pandas DataFrames).
    """
    raise NotImplementedError

def preprocess(raw_data):
    """Convert raw data into features/labels.
    Image: set up transforms (normalize, optional augmentation).
    Text: build TF-IDF (1-2 n-grams), cap vocab size.
    TODO: return (X_train, y_train, X_val, y_val, X_test, y_test) or similar.
    """
    raise NotImplementedError

def build_model():
    """Create and return your baseline model.
    Image: tiny CNN or frozen backbone + linear head.
    Text: LogisticRegression or LinearSVC.
    """
    raise NotImplementedError

def train(model, train_data, val_data=None):
    """Train the model. Track loss/metrics per epoch if applicable.
    Return the fitted model and any logs you want to plot.
    """
    raise NotImplementedError

def evaluate(model, test_data):
    """Compute final metrics.
    Classification: accuracy, precision, recall, F1, confusion matrix; PR curve optional.
    Return a dict with metrics for easy saving/printing.
    """
    raise NotImplementedError

def report(metrics):
    """Pretty-print key numbers and optionally save to JSON.
    TODO: add small tables/plots in your README using the printed output here.
    """
    print(metrics)

def main():
    # 1) Load
    raw = load_data()
    # 2) Preprocess
    data = preprocess(raw)
    # 3) Build
    model = build_model()
    # 4) Train
    model, logs = train(model, (data[0], data[1]), (data[2], data[3]))
    # 5) Evaluate
    metrics = evaluate(model, (data[4], data[5]))
    # 6) Report
    report(metrics)

if __name__ == "__main__":
    main()
