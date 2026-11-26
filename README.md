# Milk Quality Prediction: XGBoost vs AdaBoost

A reproducible machine-learning classification project comparing **XGBoost** and **AdaBoost** for predicting milk quality (Low, Medium, High) from physicochemical and sensory attributes.

> **Research integrity:** this repository does not claim a pre-computed accuracy. Run the pipeline on the dataset to generate metrics from a reproducible train/test split. Results depend on the dataset version and methodology.

## Research objective
Evaluate whether XGBoost provides better predictive performance than AdaBoost on the same milk-quality dataset, using identical partitions and multiple classification metrics.

## Dataset
Public [Milk Quality dataset on Kaggle](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality). The commonly distributed file has 1,059 observations and 7 input attributes: pH, temperature (sometimes misspelled `Temprature`), taste, odor, fat, turbidity and colour/color. The target `Grade` has three classes: Low, Medium and High. These dataset attributes and labels are described in the dataset documentation. [Dataset feature description](https://github.com/shubhamoypaul-svmy/Milk-Quality-Prediction) · [Related research description](https://dergipark.org.tr/en/download/article-file/2673216).

**Important data-quality note:** public copies may contain many repeated rows. The training pipeline removes exact duplicate records before splitting to reduce the risk of identical observations appearing in both training and test sets. It reports the remaining row count. This can substantially reduce the effective sample size, so interpret metrics cautiously and disclose this choice in academic reports.

The dataset is not redistributed. Download it from Kaggle and place the CSV at `data/raw/milknew.csv`.

## Project structure
```
.
├── data/
│   ├── raw/                  # Put milknew.csv here (not committed)
│   └── README.md
├── models/                   # Trained model artifacts (generated)
├── reports/
│   ├── figures/              # Generated EDA and evaluation charts
│   └── metrics/              # Generated metric tables/reports
├── src/
│   ├── eda.py                # Exploratory data analysis and plots
│   ├── train.py              # Preprocessing, training, evaluation
│   └── predict.py            # Prediction for a new sample
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup
Python 3.10+ recommended.

```bash
git clone https://github.com/192110399/Improving-accurancy-in-Modeling-Milk-Quality-using-XGBoost-Comparing-with-ADABoost..git
cd Improving-accurancy-in-Modeling-Milk-Quality-using-XGBoost-Comparing-with-ADABoost.
python -m venv .venv
```

Activate: Windows `.venv\\Scripts\\activate`; macOS/Linux `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

## Dataset setup
1. Download the CSV from [Kaggle](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality).
2. Create `data/raw`.
3. Rename the file to `milknew.csv` and place it in that folder.

## Step 1 — Exploratory data analysis
```bash
python src/eda.py --data data/raw/milknew.csv
```
This prints shape, types, missing values, duplicate count and grade distribution. It saves grade distribution, feature histograms and a correlation heatmap under `reports/figures/`.

## Step 2 — Train and compare models
```bash
python src/train.py --data data/raw/milknew.csv
```
The pipeline:
1. Normalizes common column-name variations and validates required columns.
2. Converts predictors to numeric values and removes rows without a target.
3. Removes exact duplicate records.
4. Encodes the three target classes.
5. Uses a stratified 80/20 train/test split with random seed 42.
6. Fits median imputation on training data only.
7. Trains XGBoost and AdaBoost on the same split.
8. Calculates accuracy, macro/weighted precision, recall and F1-score.
9. Saves confusion matrices, a comparison chart, classification reports and trained pipelines.

Generated files:
- `reports/metrics/model_comparison.csv`
- `reports/metrics/classification_reports.txt`
- `reports/figures/confusion_matrix_xgboost.png`
- `reports/figures/confusion_matrix_adaboost.png`
- `reports/figures/model_comparison.png`
- `models/xgboost_pipeline.joblib`
- `models/adaboost_pipeline.joblib`
- `models/label_encoder.joblib`

## Step 3 — Predict a new sample
After training:
```bash
python src/predict.py --model xgboost
# or
python src/predict.py --model adaboost
```
Enter pH, temperature, taste, odor, fat, turbidity and colour. The binary attributes use 0/1. Check the dataset's encoding before entering values. Predictions are educational estimates, not laboratory certification or food-safety tests.

## Methodology and metrics
- **Accuracy:** correctly classified samples / all test samples.
- **Precision:** reliability of predictions for a class.
- **Recall:** fraction of actual members of a class correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Macro average:** gives each class equal weight; useful when class counts differ.
- **Weighted average:** weights class metrics by support.
- **Confusion matrix:** reveals which grades are confused with one another.

Both models use the same held-out test set. The preprocessing pipeline is fitted on training data only. The macro F1-score is used as the primary comparison metric, with accuracy and class-wise results reported alongside it.

## Research questions
1. Which algorithm achieves higher test accuracy?
2. Which model has the strongest macro F1-score across all grades?
3. Which quality grade is most frequently misclassified?
4. Do the models show different precision/recall trade-offs?

## Limitations
- The dataset is small and manually collected; generalization to real dairy production is not guaranteed.
- Exact duplicates can greatly reduce the effective dataset size when removed.
- Sensory variables are simplified binary indicators.
- Random holdout scores may vary with dataset version and split.
- Model predictions must not replace validated dairy laboratory testing.

## Suggested abstract
This project investigates milk quality classification using two ensemble learning algorithms, XGBoost and AdaBoost. The dataset contains pH, temperature, taste, odor, fat, turbidity and colour measurements, with milk grade categorized as low, medium or high. The workflow includes data validation, exploratory analysis, duplicate handling, target encoding and a stratified train-test split. Both algorithms are evaluated on the same held-out data using accuracy, precision, recall, macro F1-score and confusion matrices. The experimental results generated by the supplied pipeline are used to identify the stronger model and analyze class-wise errors. The study provides a reproducible comparison of boosting methods for an educational milk-quality prediction task.

## Author
**Gokul S.** — Computer Science Engineering

## License
Source code is provided under the MIT License. The dataset remains subject to the original provider's terms; download and use it accordingly.
