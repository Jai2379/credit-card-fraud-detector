# Credit Card Fraud Detector

Random Forest fraud classifier with a drift-triggered retraining loop. The model is built on the 2013 European cardholder dataset, then validated against a newer dataset. It rebuilds itself when recall on the new data falls below 0.90 or when the features it was trained on no longer exist in the incoming schema.

The retraining pipeline object is named `mahoraga`, after the shikigami that adapts to whatever hits it.

## Dataset profile

From `fraud.ipynb`, on `creditcard.csv` (2013):

| Property | Value |
|:--|:--|
| Transactions | 284,807 |
| Columns | 31: `Time`, `V1`–`V28` (PCA components), `Amount`, `Class` |
| Fraud cases | 492 (0.173%) |
| Mean amount, genuine / fraud | 88.29 / 122.21 |
| Median amount, genuine / fraud | 22.00 / 9.25 |
| Strongest negative correlation with `Class` | `V17` −0.326, `V14` −0.303, `V12` −0.261, `V10` −0.217, `V16` −0.197 |
| Strongest positive correlation with `Class` | `V11` +0.155, `V4` +0.133 |

Fraud has a higher mean but a lower median amount than genuine traffic, so fraudulent amounts are the more right-skewed of the two: many small charges plus a tail of large ones.

## Architecture

```
creditcard_2023.csv
 │
 ├── importer.py ──► fraud_detection.db (SQLite, table `transactions`)
 │                          │
 │                          └──► live_check.py   random row → check_transaction()
 │
 └── final_test.py
       ├── no model on disk        ──► train
       ├── manifest features missing ──► retrain     (schema drift)
       ├── recall < 0.90            ──► retrain     (concept drift)
       └── recall ≥ 0.90            ──► keep current model
                    │
                    ▼
       pipeline.py  FraudPipeline.auto_retrain()  ──►  models/active_model.pkl
                                                            │
                                         predict_fraud.check_transaction()
```

## Components

**`pipeline.py`: `FraudPipeline.auto_retrain(data_path)`**

1. Target discovery: the last column with exactly two unique values whose name does not contain `id`.
2. Candidate features: numeric columns minus the target and `id`, `rowid`, `time`, `unnamed: 0`.
3. `SelectKBest(f_classif, k=min(n_features, 6))`.
4. `SMOTE(random_state=42)` to balance classes.
5. `RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)`.
6. Serialize `{"model", "features", "target_name"}` to `models/active_model.pkl`.

Because the target and features are discovered at runtime, a new dataset with a different column set needs no code change.

**`final_test.py`: `run_mahoraga_verification(model_path, data_path)`**

Loads the active model, checks that every manifest feature exists in the new file, scores recall with `sklearn.metrics.recall_score`, and retrains if either check fails. After a rebuild it reloads the new artifact and prints its recall on the same file.

**`predict_fraud.py`: `check_transaction(tx: dict) -> str`**

Pads missing manifest features with `0.0`, selects columns in manifest order, prints `P(fraud)` from `predict_proba` and returns an alert or clear string. The manifest travels with the model, so the serving input always matches the training feature set.

**`live_check.py`**

Finds the first table through `sqlite_master`, pulls one transaction with `ORDER BY RANDOM() LIMIT 1`, strips `rowid` and scores it.

**`analysis.py`**

Baseline experiment on the 2013 data. Loads the CSV into SQLite, selects `V17, V14, V12, V10, V4, Amount` with a SQL query, applies SMOTE, trains a Random Forest (`max_depth=10`, `min_samples_leaf=10`) and prints a classification report. Writes `final_fraud_detector.pkl`.

## Setup

```bash
git clone https://github.com/Jai2379/credit-card-fraud-detector.git
cd credit-card-fraud-detector
pip install -r requirements.txt
```

Place both datasets in the repo root:

- `creditcard.csv`: [Credit Card Fraud Detection (2013)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- `creditcard_2023.csv`: [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

Any replacement dataset needs numeric features and one binary label column.

```bash
python importer.py       # creditcard_2023.csv → fraud_detection.db
python final_test.py     # verify active model, retrain on drift
python live_check.py     # score a random stored transaction
```

## Layout

```
.
├── analysis.py          baseline SQL + SMOTE + Random Forest experiment
├── fraud.ipynb          EDA: class balance, amount distribution, fraud by hour, correlations
├── importer.py          CSV → SQLite
├── pipeline.py          FraudPipeline / mahoraga
├── final_test.py        drift check and retrain trigger
├── predict_fraud.py     check_transaction()
├── live_check.py        random SQLite row → inference
├── models/active_model.pkl
└── requirements.txt
```

## Known gaps

- `analysis.py` applies SMOTE before the train/test split, so synthetic samples leak into the test set and its report is optimistic.
- After a retrain, `final_test.py` measures recall on the data it just trained on.
- `creditcard_2023.csv` is class-balanced while the 2013 set is 0.17% fraud, so recall on the two is not directly comparable.
- Target discovery picks the last binary column and will misfire if a binary feature appears after the label.
