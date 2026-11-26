"""Predict milk quality using a previously trained model pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["ph", "temperature", "taste", "odor", "fat", "turbidity", "colour"]


def ask_number(label: str, binary: bool = False) -> float:
    while True:
        raw = input(f"{label}: ").strip()
        try:
            value = float(raw)
            if binary and value not in (0.0, 1.0):
                print("Please enter 0 or 1.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["xgboost", "adaboost"], default="xgboost",
        help="Which trained model to use."
    )
    args = parser.parse_args()

    model_path = ROOT / "models" / f"{args.model}_pipeline.joblib"
    encoder_path = ROOT / "models" / "label_encoder.joblib"
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            "Trained files were not found. Run src/train.py before prediction."
        )

    print("\nEnter milk sample characteristics")
    values = {
        "ph": ask_number("pH"),
        "temperature": ask_number("Temperature (C)"),
        "taste": ask_number("Taste (0 = poor, 1 = good)", binary=True),
        "odor": ask_number("Odor (0 = poor, 1 = good)", binary=True),
        "fat": ask_number("Fat (0 = low, 1 = high)", binary=True),
        "turbidity": ask_number("Turbidity (0 = low, 1 = high)", binary=True),
        "colour": ask_number("Colour (dataset numeric code)"),
    }

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    sample = pd.DataFrame([[values[column] for column in FEATURES]], columns=FEATURES)
    predicted_code = int(model.predict(sample)[0])
    predicted_grade = encoder.inverse_transform([predicted_code])[0]

    print(f"\nPredicted milk grade ({args.model}): {str(predicted_grade).title()}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]
        print("Class probabilities:")
        for label, probability in zip(encoder.classes_, probabilities):
            print(f"  {str(label).title():<10} {probability:.2%}")
    print("\nNote: This is an educational model prediction, not a food-safety test.")


if __name__ == "__main__":
    main()
