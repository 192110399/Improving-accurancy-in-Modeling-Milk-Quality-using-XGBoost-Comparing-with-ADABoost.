"""Train and compare XGBoost and AdaBoost for milk quality classification."""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["ph", "temperature", "taste", "odor", "fat", "turbidity", "colour"]
TARGET = "grade"
RANDOM_STATE = 42


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common dataset spelling/case variations."""
    frame = frame.copy()
    frame.columns = (
        frame.columns.astype(str).str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "", regex=True)
    )
    aliases = {
        "temprature": "temperature",
        "temp": "temperature",
        "color": "colour",
        "phvalue": "ph",
        "odor": "odor",
        "fat": "fat",
        "grade": "grade",
    }
    frame = frame.rename(columns={c: aliases.get(c, c) for c in frame.columns})
    return frame


def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Download the Kaggle Milk Quality CSV and place it at data/raw/milknew.csv."
        )

    data = normalize_columns(pd.read_csv(csv_path))
    missing = [c for c in FEATURES + [TARGET] if c not in data.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(data.columns)}"
        )

    data = data[FEATURES + [TARGET]].copy()
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[TARGET] = data[TARGET].astype("string").str.strip().str.lower()
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET] != ""]
    data = data.drop_duplicates().reset_index(drop=True)

    if data[TARGET].nunique() < 2:
        raise ValueError("The target column must contain at least two quality classes.")
    if data.empty:
        raise ValueError("No usable rows remain after cleaning.")

    return data


def build_models(class_count: int):
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=class_count,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    ada = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.5,
        random_state=RANDOM_STATE,
    )
    return {
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", xgb),
        ]),
        "AdaBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ada),
        ]),
    }


def evaluate_model(name, model, x_test, y_test, class_names):
    predictions = model.predict(x_test)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
    )
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1-score (Macro)": f1_macro,
        "Precision (Weighted)": precision_weighted,
        "Recall (Weighted)": recall_weighted,
        "F1-score (Weighted)": f1_weighted,
    }
    report = classification_report(
        y_test, predictions, labels=np.arange(len(class_names)),
        target_names=class_names, zero_division=0
    )
    matrix = confusion_matrix(
        y_test, predictions, labels=np.arange(len(class_names))
    )
    return metrics, report, matrix


def save_confusion_matrix(matrix, class_names, title, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "raw" / "milknew.csv",
        help="Path to the milk-quality CSV file."
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    data = load_dataset(args.data)

    print("\nDATASET OVERVIEW")
    print(f"Rows after removing exact duplicates: {len(data)}")
    print(f"Duplicate rows in original file: {pd.read_csv(args.data).duplicated().sum()}")
    print("\nClass distribution:")
    print(data[TARGET].value_counts())

    x = data[FEATURES]
    encoder = LabelEncoder()
    y = encoder.fit_transform(data[TARGET].astype(str))
    class_names = [str(label) for label in encoder.classes_]

    counts = pd.Series(y).value_counts()
    if counts.min() < 2:
        raise ValueError(
            "Each target class needs at least two samples for a stratified "
            "train/test split. Check the dataset and class distribution."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    models = build_models(class_count=len(class_names))
    reports_dir = ROOT / "reports" / "metrics"
    figures_dir = ROOT / "reports" / "figures"
    models_dir = ROOT / "models"
    for folder in (reports_dir, figures_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    results = []
    report_text = [
        "MILK QUALITY CLASSIFICATION: MODEL EVALUATION",
        f"Rows used: {len(data)}",
        f"Train rows: {len(x_train)} | Test rows: {len(x_test)}",
        f"Classes: {', '.join(class_names)}",
        f"Random state: {RANDOM_STATE}",
        "\nNote: metrics are calculated on the held-out test set.",
    ]

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        metrics, report, matrix = evaluate_model(
            name, model, x_test, y_test, class_names
        )
        results.append(metrics)
        report_text.extend([f"\n{'=' * 60}", name, report])
        save_confusion_matrix(
            matrix, class_names, f"{name} Confusion Matrix",
            figures_dir / f"confusion_matrix_{name.lower()}.png"
        )
        joblib.dump(model, models_dir / f"{name.lower()}_pipeline.joblib")
        print(f"Test accuracy: {metrics['Accuracy']:.4f}")
        print(report)

    comparison = pd.DataFrame(results).sort_values(
        by="F1-score (Macro)", ascending=False
    )
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "classification_reports.txt").write_text(
        "\n".join(report_text), encoding="utf-8"
    )
    joblib.dump(encoder, models_dir / "label_encoder.joblib")

    plot_data = comparison.set_index("Model")[
        ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-score (Macro)"]
    ]
    ax = plot_data.plot(kind="bar", figsize=(10, 6), ylim=(0, 1), rot=0)
    ax.set_title("XGBoost vs AdaBoost - Test Set Performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=160)
    plt.close()

    print("\nFINAL MODEL COMPARISON")
    print(comparison.to_string(index=False))
    print("\nSaved metrics, figures and model pipelines in the project folders.")
    print("Best model by macro F1:", comparison.iloc[0]["Model"])


if __name__ == "__main__":
    main()
