import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb





def load_data(train_path: str = "train.csv",
              test_path: str = "test.csv"):
    """
    Load the Kaggle Playground S5E11 train and test CSV files.

    Parameters
    ----------
    train_path : str
        File path for train.csv
    test_path : str
        File path for test.csv

    Returns
    -------
    train : pd.DataFrame
        Training data with features + target (loan_paid_back)
    test : pd.DataFrame
        Test data with only features (no target)
    """
    # Read CSVs into DataFrames
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Basic info to understand the data
    print("=== DATA LOADED ===")
    print(f"Train shape: {train.shape}")  # (rows, columns)
    print(f"Test shape:  {test.shape}\n")

    print("=== TRAIN HEAD ===")
    print(train.head(), "\n")  # first 5 rows

    print("=== TRAIN INFO ===")
    print(train.info())        # column types, null counts

    return train, test

def basic_cleaning(train: pd.DataFrame,
                   test: pd.DataFrame,
                   target_col: str = "loan_paid_back"):
    """
    Perform basic cleaning:
    - Remove duplicate rows in train
    - Combine train & test so cleaning is consistent
    - Fill missing numeric values with median
    - Fill missing categorical values with 'Unknown'
    """

    # --- 1. Remove duplicate rows ---
    before = train.shape[0]
    train = train.drop_duplicates()
    after = train.shape[0]
    print(f"\nRemoved {before - after} duplicate rows from train")

    # --- 2. Tag train/test so we can split later ---
    train["is_train"] = 1
    test["is_train"] = 0

    # Combine them
    full = pd.concat([train, test], ignore_index=True)

    # --- 3. Separate numeric + categorical columns ---
    numeric_cols = full.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = full.select_dtypes(include=["object", "category"]).columns.tolist()

    # Don't fill the target if it's numeric
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # --- 4. Fill missing values ---
    for col in numeric_cols:
        full[col] = full[col].fillna(full[col].median())

    for col in cat_cols:
        full[col] = full[col].fillna("Unknown")

    print("\nMissing values handled.")
    return full

def inspect_sample_submission(path: str = "sample_submission.csv"):
    """
    Inspect Kaggle's sample_submission.csv to understand:
    - column names
    - example values
    So we can see if the target column looks like 0/1 labels or probabilities.
    """
    print("\n=== Inspecting sample_submission.csv ===")
    df = pd.read_csv(path)

    print("\nShape:", df.shape)
    print("\nHead:")
    print(df.head())

    print("\nInfo:")
    print(df.info())

    # Show basic stats on the target column (2nd column)
    if df.shape[1] >= 2:
        target_col = df.columns[1]
        print(f"\nTarget column name in sample_submission: {target_col}")

        print("\nValue counts (first 10):")
        print(df[target_col].value_counts().head(10))

        print("\nDescribe():")
        print(df[target_col].describe())
    else:
        print("\n[Warning] sample_submission has less than 2 columns? Very unusual.")

    return df


def encode_categoricals(full: pd.DataFrame,
                        target_col: str = "loan_paid_back"):
    """
    Convert categorical (object) columns into numeric labels.
    We use LabelEncoder for each column separately.
    """

    # --- 1. Identify categorical columns ---
    cat_cols = full.select_dtypes(include=["object", "category"]).columns.tolist()

    # Do NOT encode the target if it's object
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # To store encoders so we can reuse later if needed
    label_encoders = {}

    # --- 2. Encode each categorical column ---
    for col in cat_cols:
        le = LabelEncoder()
        full[col] = le.fit_transform(full[col])
        label_encoders[col] = le

    print("\nCategorical columns encoded.")
    return full, label_encoders

def split_full_to_data(full: pd.DataFrame,
                       target_col: str = "loan_paid_back"):
    """
    Split the combined 'full' DataFrame back into:
    - X, y for training
    - X_test and test_ids for submission
    using the 'is_train' flag.
    """

    # --- 1. Split back into train and test using is_train ---
    train_clean = full[full["is_train"] == 1].copy()
    test_clean = full[full["is_train"] == 0].copy()

    # --- 2. Drop the helper column 'is_train' ---
    train_clean = train_clean.drop(columns=["is_train"])
    test_clean = test_clean.drop(columns=["is_train"])

    # --- 3. Separate features (X) and target (y) from train ---
    y = train_clean[target_col]
    X = train_clean.drop(columns=[target_col, "id"], errors="ignore")


    # --- 4. For test data, keep id separately and drop id from features ---
    test_ids = test_clean["id"]
    # test also has target_col column but only NaNs; we drop it & id
    X_test = test_clean.drop(columns=[target_col, "id"], errors="ignore")

    return X, y, X_test, test_ids

def train_and_validate_model(X, y):
    """
    Split X, y into train/validation sets,
    train a RandomForest model, and print ROC-AUC on validation data.
    """

    # --- 1. Split into train and validation sets ---
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,        # 20% for validation, 80% for training
        random_state=42,      # for reproducibility
        stratify=y            # keep class balance (0/1) same in both sets
    )

    print("\nTrain/Val shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    # --- 2. Define the model ---
    model = RandomForestClassifier(
        n_estimators=300,     # number of trees
        max_depth=None,       # let trees grow until pure or min samples
        n_jobs=-1,            # use all CPU cores
        random_state=42
    )

    # --- 3. Train the model ---
    print("\nTraining RandomForest...")
    model.fit(X_train, y_train)

    # --- 4. Evaluate using ROC-AUC on validation set ---
    # predict_proba gives probability for both classes [P(class0), P(class1)]
    y_val_proba = model.predict_proba(X_val)[:, 1]  # take probability of class 1

    auc = roc_auc_score(y_val, y_val_proba)
    print(f"\nValidation ROC-AUC: {auc:.4f}")

    return model

def train_on_full_and_predict(model, X, y, X_test):
    """
    Retrain the model on the FULL training data (X, y),
    then predict probabilities for X_test.
    """

    print("\nTraining model on FULL data...")
    model.fit(X, y)

    print("Creating test predictions...")
    test_proba = model.predict_proba(X_test)[:, 1]

    return test_proba

def create_submission(test_ids, test_proba, filename="submission.csv"):
    """
    Combine test_ids and predicted probabilities into a Kaggle submission file.
    """

    submission = pd.DataFrame({
        "id": test_ids,
        "loan_paid_back": test_proba
    })

    submission.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")

def train_and_validate_lgbm(X, y):
    """
    Train a strong LightGBM model with train/validation split.
    Returns the trained model.
    """

    # 1. Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining LightGBM...")

    # 2. Create dataset format for LGB
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # 3. LightGBM hyperparameters (strong defaults)
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.03,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 2,
        "verbose": -1,
        "seed": 42
    }

    # 4. Train the model with early stopping
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]
    )


    return model

def predict_lgbm(model, X_test):
    print("\nPredicting with LightGBM...")
    test_proba = model.predict(X_test)
    return test_proba


def main():
    # Call the loader function
    train, test = load_data("train.csv", "test.csv")

        # Optional: inspect sample_submission to understand expected output format
    inspect_sample_submission("sample_submission.csv")
    

    # STEP 2: Basic cleaning
    full = basic_cleaning(train, test, target_col="loan_paid_back")
    # ADD THESE TWO LINES
    full, encoders = encode_categoricals(full, target_col="loan_paid_back")

    print("\n=== AFTER CLEANING ===")
    print(full.info())
    print(full.head())

    X, y, X_test, test_ids = split_full_to_data(full, target_col="loan_paid_back")

    print("\nShapes after split:")
    print("X:", X.shape)
    print("y:", y.shape)
    print("X_test:", X_test.shape)
    print("test_ids:", test_ids.shape)

    model = train_and_validate_lgbm(X, y)
    test_proba = predict_lgbm(model, X_test)


    # STEP 7: Create submission file
    create_submission(test_ids, test_proba, filename="submission_final.csv")







if __name__ == "__main__":
    main()
