"""Exploratory data analysis for the milk quality dataset."""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["ph", "temperature", "taste", "odor", "fat", "turbidity", "colour"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/milknew.csv")
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    df = pd.read_csv(args.data)
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
    df = df.rename(columns={"temprature": "temperature", "color": "colour"})
    missing = [c for c in FEATURES + ["grade"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nExact duplicate rows:", df.duplicated().sum())
    print("\nGrade distribution:\n", df["grade"].value_counts())
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="grade", order=df["grade"].value_counts().index, ax=ax)
    ax.set_title("Milk quality grade distribution")
    fig.tight_layout()
    fig.savefig(out / "grade_distribution.png", dpi=160)
    plt.close(fig)
    numeric = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    numeric.hist(figsize=(12, 9), bins=20)
    plt.suptitle("Feature distributions")
    plt.tight_layout()
    plt.savefig(out / "feature_distributions.png", dpi=160)
    plt.close()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=160)
    plt.close(fig)
    print(f"EDA charts saved to {out}")

if __name__ == "__main__":
    main()
