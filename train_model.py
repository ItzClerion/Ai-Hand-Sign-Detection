"""
train_model.py

Reads all CSV files from the dataset/ folder (one per label, e.g. A.csv, B.csv),
trains a RandomForest classifier on the hand landmark features, and saves the
trained model + label list for use in live prediction.

Usage:
    python train_model.py

Requires: dataset/<LABEL>.csv files created by collect_data.py
Outputs:
    model/sign_language_model.pkl
    model/labels.json
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_dataset(dataset_dir="dataset"):
    csv_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{dataset_dir}/'. Run collect_data.py first."
        )

    X_list = []
    y_list = []

    for csv_path in csv_files:
        label = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"WARNING: {csv_path} is empty, skipping.")
            continue

        X_list.append(df.values)
        y_list.extend([label] * len(df))
        print(f"Loaded {len(df)} samples for label '{label}'")

    if not X_list:
        raise ValueError("No usable data found in any CSV file.")

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"\nTotal samples: {len(X)}, Feature size: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nValidation accuracy: {acc * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, os.path.join("model", "sign_language_model.pkl"))

    labels = sorted(set(y))
    with open(os.path.join("model", "labels.json"), "w") as f:
        json.dump(labels, f)

    print(f"\nModel saved to model/sign_language_model.pkl")
    print(f"Labels saved to model/labels.json -> {labels}")


if __name__ == "__main__":
    main()