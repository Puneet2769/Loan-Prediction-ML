# 💳 Loan Repayment — Playground S5E11 Pipeline  
### LightGBM + RandomForest baseline for the Kaggle Playground S5E11 task

End-to-end pipeline that loads data, cleans and encodes features, trains a strong LightGBM model (with a RandomForest fallback pattern available), evaluates with ROC-AUC, and writes a Kaggle-ready submission. Built for the Kaggle Playground S5E11 setup.

---

## 📘 Competition / Context  
**Kaggle Playground:** Playground S5E11 https://www.kaggle.com/competitions/playground-series-s5e11  
**Expected files:**  
- `train.csv` → training data with target `loan_paid_back`  
- `test.csv` → test data for submission  
- `sample_submission.csv` → used to inspect expected output format  
*(Datasets not included — download and place them in the repo root.)*

---

## ⚙️ What this Project Does (overview)

1. Loads `train.csv` and `test.csv`, prints basic shapes & info.  
2. Basic cleaning:
   - Drops duplicate rows.  
   - Combines train/test to apply consistent cleaning.  
   - Fills numeric missing values with median.  
   - Fills categorical missing values with `"Unknown"`.  
3. Encodes categorical features using `LabelEncoder` (per column).  
4. Splits back into `X, y` and `X_test` (keeps `id` for submission).  
5. Trains a LightGBM model with early stopping and evaluates with ROC-AUC.  
6. Predicts probabilities for test set and saves `submission_final.csv` with columns `id, loan_paid_back`.

---

## 🧠 Model & Pipeline Notes

- **Primary model:** LightGBM (`lgb.train`) with:
  - `objective: binary`, `metric: auc`, `learning_rate: 0.03`, `num_leaves: 64`
  - early stopping and sensible feature/bagging fractions
- **Baseline classifier option:** RandomForestClassifier (used in earlier EDA / baseline steps)
- **Evaluation metric:** ROC-AUC (validation)
- **Submission:** probabilities for positive class (`loan_paid_back`) — matches `sample_submission.csv` format

---

## 🚀 Quick Start — Run locally

Save the script as `playground_s5e11_pipeline.py` (or keep your filename), place required CSVs in the repo root, then run:

```bash
python playground_s5e11_pipeline.py
```

What the script does:

Loads and prints dataset info.

Cleans and encodes features.

Trains LightGBM with early stopping and prints validation AUC.

Predicts test probabilities and writes submission_final.csv.

📁 Repository structure
bash
Copy code
├── playground_s5e11_pipeline.py   # main script (the code you shared)
├── submission_final.csv           # generated submission (after running)
├── train.csv                      # (not included)
├── test.csv                       # (not included)
├── sample_submission.csv          # (not included)
├── requirements.txt
└── README.md
✅ Quick improvements & next steps
Use cross-validated target encoding for high-cardinality categoricals.

Try KFold CV + out-of-fold predictions for stacking / ensembling.

Experiment with LightGBM hyperparameter tuning (num_leaves, min_data_in_leaf, learning rate).

Add calibration step if the submission expects well-calibrated probabilities.

Save fitted LabelEncoders (or use sklearn.pipeline + OrdinalEncoder) so train/test encoding is reproducible.

# 👤 Author
Puneet Poddar
Kaggle: https://www.kaggle.com/puneet2769

# 📌 License / Attribution
Use freely for experiments and learning. When publishing, credit the original dataset source (Kaggle).
