# Milk Quality Prediction: XGBoost vs AdaBoost

A reproducible machine-learning classification project that compares **XGBoost** and **AdaBoost** for predicting milk quality (Low, Medium, High) from physicochemical and sensory attributes.

> **Research integrity:** this repository does not claim a pre-computed accuracy. Run the pipeline on the downloaded dataset to generate metrics from a reproducible train/test split. Results depend on the dataset version and environment.

## Research objective
To evaluate whether gradient-boosted trees (XGBoost) provide better predictive performance than AdaBoost on the same milk-quality dataset, using identical train/test partitions and multiple classification metrics.

## Dataset
This project is designed for the public [Milk Quality dataset on Kaggle](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality). The commonly distributed file contains 1,059 observations and columns similar to:

- Inputs: `pH`, `Temprature` (or `Temperature`), `Taste`, `Odor`, `Fat`, `Turbidity`, `Colour` (or `Color`)
- Target: `Grade` (Low, Medium, High)

Dataset column names can vary slightly between copies. The training script normalizes common spelling variations. **The dataset is not redistributed here**; download it from the source and place the CSV at `data/raw/milknew.csv`.

## Project structure
```
.
├── data/
│   ├── raw/                  # Put milknew.csv here (not committed)
│   └── README.md
├── models/                   # Trained model artifacts (generated)
├── reports/
│   ├── figures/              # Generated charts
│   └── metrics/              # Generated metric tables/reports
├── src/
│   ├── train.py              # Preprocessing, training, evaluation, charts
│   └── predict.py            # Predict a new milk sample using saved model
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup
Python 3.10 or newer is recommended.

```bash
git clone https://github.com/192110399/Improving-accurancy-in-Modeling-Milk-Quality-using-XGBoost-Comparing-with-ADABoost..git
cd Improving-accurancy-in-Modeling-Milk-Quality-using-XGBoost-Comparing-with-ADABoost.
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Download and place the dataset
1. Download the CSV from the Kaggle dataset page.
2. Create the folder `data/raw`.
3. Rename the downloaded CSV to `milknew.csv`.
4. Place it at `data/raw/milknew.csv`.

## Train and compare
Run from the repository root:

```bash
python src/train.py --data data/raw/milknew.csv
```

The script:
1. Checks required columns, missing values, duplicates and class counts.
2. Normalizes common column-name variations.
3. Uses a stratified 80/20 train/test split with a fixed random seed.
4. Fits preprocessing only on training data to avoid leakage.
5. Trains XGBoost and AdaBoost on the same split.
6. Reports accuracy, macro/weighted precision, recall and F1-score.
7. Saves confusion matrices, a metric comparison chart, classification reports and trained pipelines.

Generated outputs:
- `reports/metrics/model_comparison.csv`
- `reports/metrics/classification_reports.txt`
- `reports/figures/confusion_matrix_xgboost.png`
- `reports/figures/confusion_matrix_adaboost.png`
- `reports/figures/model_comparison.png`
- `models/xgboost_pipeline.joblib`
- `models/adaboost_pipeline.joblib`
- `models/label_encoder.joblib`

## Predict a new sample
After training, run:

```bash
python src/predict.py --model xgboost
```

Enter pH, temperature, taste, odor, fat, turbidity and colour when prompted. Use 0/1 for binary inputs. The model output is a prediction based on the training dataset; it is not a laboratory certification or food-safety test.

## Evaluation methodology
- **Accuracy:** fraction of correct predictions.
- **Precision:** reliability of positive predictions, reported macro and weighted.
- **Recall:** proportion of actual class members correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** class-level error analysis.

Both models use the same holdout data. A fixed random seed makes the split reproducible. The preprocessing pipeline is fitted on training data only. No oversampling is applied to the test set. Because this is a relatively small, manually collected dataset, report class-wise metrics and interpret very high scores cautiously; duplicate or near-duplicate records can make random holdout estimates optimistic.

## Research questions
1. Which algorithm achieves higher test accuracy?
2. Which model provides the strongest macro F1-score across all grades?
3. Which quality grade is most frequently misclassified?
4. Do the two algorithms show different precision/recall trade-offs?

## Limitations
- Dataset size and collection method may limit generalization.
- Sensory attributes are represented as simplified numeric/binary variables.
- Predictions reflect dataset patterns and should not replace validated dairy laboratory testing.
- Results are not guaranteed to match other published implementations because splits, preprocessing and duplicate handling differ.

## Author
**Gokul S.**  
Computer Science Engineering

## License
This repository's source code is provided under the MIT License. The dataset remains subject to its original provider's terms; download and use it in accordance with those terms.
