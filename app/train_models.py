"""Fit both models the UI serves and save them to models/.

    python app/train_models.py

1. text model  - the notebook's soft-vote (LR + Extra Trees) over caption /
                 hashtag / metadata features, refit on all 452 labelled posts.
2. image model - colour + texture descriptor over the 452 labelled JPEGs, for
                 uploads that are not in the dataset.

Both are cross-validated before the final fit so the UI can quote an honest
accuracy instead of a training-set number.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import cross_validate, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import pavbhaji_lib as L
from image_model import extract_features, image_model

ROOT = L.ROOT
MODELDIR = os.path.join(ROOT, "models")
os.makedirs(MODELDIR, exist_ok=True)

SCORING = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
CV = StratifiedKFold(5, shuffle=True, random_state=42)


def train_text(data_dir):
    print("\n=== text model (caption / hashtags / metadata) ===")
    raw = L.load_dataset(data_dir)
    feat = L.build_features(raw)
    lab = feat[feat["label"].notna()].reset_index(drop=True).copy()
    lab["label"] = lab["label"].astype(int)
    y = lab["label"].values
    print(f"labelled posts: {len(lab)}  ({int(y.sum())} pav bhaji / {int((1-y).sum())} not)")

    model = L.text_model()
    t0 = time.time()
    cv = cross_validate(model, lab, y, cv=CV, scoring=SCORING, n_jobs=-1)
    metrics = {k: float(cv[f"test_{k}"].mean()) for k in SCORING}
    print(f"5-fold CV: accuracy {metrics['accuracy']:.3f} | "
          f"f1 {metrics['f1']:.3f} | roc_auc {metrics['roc_auc']:.3f} "
          f"({time.time()-t0:.0f}s)")

    model.fit(lab, y)
    dump(model, os.path.join(MODELDIR, "text_model.joblib"))

    # lookup table: image filename -> the post that image belongs to
    keep = ["id", "shortcode", "image_file", "label", "caption", "tags", "likes",
            "comments", "owner_id", "width", "height", "timestamp",
            "has_location", "location_name", "is_video", "comments_disabled"]
    feat[keep].to_pickle(os.path.join(MODELDIR, "posts.pkl"))
    print(f"saved text_model.joblib and posts.pkl ({len(feat)} posts indexed)")
    return metrics


def train_image(data_dir):
    print("\n=== image model (pixels) ===")
    X, y, files = [], [], []
    for lbl in (0, 1):
        folder = os.path.join(data_dir, "images", str(lbl))
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                X.append(extract_features(os.path.join(folder, fn)))
            except Exception as e:                      # corrupt jpeg
                print("  skipped", fn, type(e).__name__)
                continue
            y.append(lbl)
            files.append(fn)
            if len(files) % 100 == 0:
                print(f"  featurised {len(files)} images")
    X = np.vstack(X)
    y = np.asarray(y)
    print(f"feature matrix: {X.shape}  ({int(y.sum())} pav bhaji / {int((1-y).sum())} not)")

    model = image_model()
    t0 = time.time()
    cv = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    metrics = {k: float(cv[f"test_{k}"].mean()) for k in SCORING}
    metrics["sd_accuracy"] = float(cv["test_accuracy"].std())
    print(f"5-fold CV: accuracy {metrics['accuracy']:.3f} "
          f"+/- {metrics['sd_accuracy']:.3f} | f1 {metrics['f1']:.3f} | "
          f"roc_auc {metrics['roc_auc']:.3f} ({time.time()-t0:.0f}s)")
    print(f"majority baseline: {max(y.mean(), 1-y.mean()):.3f}")

    model.fit(X, y)
    dump(model, os.path.join(MODELDIR, "image_model.joblib"))
    print("saved image_model.joblib")
    return metrics


def main():
    data_dir = L.find_data_dir()
    print("dataset:", os.path.abspath(data_dir))
    text_metrics = train_text(data_dir)
    image_metrics = train_image(data_dir)

    with open(os.path.join(MODELDIR, "metrics.json"), "w") as fh:
        json.dump({"text": text_metrics, "image": image_metrics}, fh, indent=2)
    print("\nsaved models/metrics.json - done")


if __name__ == "__main__":
    main()
