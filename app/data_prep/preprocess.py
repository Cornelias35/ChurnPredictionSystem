import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb
import joblib
import os

def scale_numeric_columns(numeric_columns_, x_train_, x_test_ = None, ):

    scaler_path = "app/paths/scaler.pkl"

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(x_train_[numeric_columns_])
        joblib.dump(scaler, scaler_path)
    x_train_[numeric_columns_] = scaler.transform(x_train_[numeric_columns_])

    if x_test_ is not None:
        x_test_[numeric_columns_] = scaler.transform(x_test_[numeric_columns_])
        return x_train_, x_test_
    return x_train_

def transforming_data(df, binary_cols, service_cols, multi_cols):
    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))
    df[service_cols] = df[service_cols].apply(
        lambda x: x.map({"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}))
    df = pd.get_dummies(df, columns=multi_cols, dtype=np.int8)
    df.columns = [col.replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]

    return df

def prepare_train_dataset(test_size_=0.2, random_state_=42):
    """
    Prepare the Telco Customer Churn dataset for model training.

    Assumptions about the input data:
    - Input file: '../data/train.csv'
    - Binary columns: Partner, Dependents, PhoneService, PaperlessBilling, Churn
      (encoded as Yes=1, No=0).
    - Service columns: MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection,
      TechSupport, StreamingTV, StreamingMovies
      (encoded as Yes=1, No=0, "No internet service"/"No phone service"=0).
    - Categorical columns: gender, InternetService, Contract, PaymentMethod
      (encoded using OneHotEncoder via pandas.get_dummies, with drop_first=True
      to avoid multicollinearity).
    - Null values are only expected in 'TotalCharges', which are dropped.

    Preprocessing steps:
    1. Read dataset from CSV.
    2. Convert 'TotalCharges' to numeric and drop rows with missing values.
    3. Encode binary, service, and categorical columns.
    4. Split into train/test sets.
    5. Standardize numeric columns using StandardScaler.

    Parameters
    ----------
    test_size_ : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    random_state_ : int, optional (default=42)
        Random seed for reproducibility.

    Returns
    -------
    x_train : pd.DataFrame
        Training features after preprocessing and scaling.
    x_test : pd.DataFrame
        Test features after preprocessing and scaling.
    y_train : pd.Series
        Training labels (Churn).
    y_test : pd.Series
        Test labels (Churn).
    FileNotFoundError
        If '../data/train.csv' is not found, the function logs an error
        and raises a FileNotFoundError.
    """

    try:
        df = pd.read_csv('app/dataset/telco-customer-churn.csv')
    except FileNotFoundError:
        logging.error('Could not find train.csv')
        raise FileNotFoundError

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna(subset=["TotalCharges"])

    num_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    churn_map = {
        "No": 0,
        "Yes": 1,
    }
    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map(churn_map)

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies"]
    multi_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]

    df = transforming_data(df, binary_cols, service_cols, multi_cols)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_, random_state=random_state_)

    x_train, x_test = scale_numeric_columns(x_train_=x_train, x_test_=x_test, numeric_columns_=num_columns)

    return x_train, x_test, y_train, y_test


def log_cleaned_data(df, artifact_name="cleaned-churn-data", version="v1"):
    """
    Log the cleaned dataset to Weights & Biases as an artifact.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataset to log.
    artifact_name : str, optional
        The name of the artifact (default: "cleaned-churn-data").
    version : str, optional
        The version tag for the artifact (default: "v1").
    """
    with wandb.init(project="customer-shurn", job_type="cleaned-churn-data") as run:

        file_path = f"{artifact_name}.csv"
        df.to_csv(file_path, index=False)

        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description="Cleaned Telco Customer Churn dataset",
            metadata={"rows": df.shape[0], "columns": df.shape[1]}
        )
        artifact.add_file(file_path)

        run.log_artifact(artifact)

        churn_table = wandb.Table(dataframe=df.head(200))
        run.log({"cleaned_churn_preview": churn_table})


def load_dataset(test_size_=0.2, random_state_=42, artifact_name="cleaned-churn-data"):
    """
    Load the cleaned dataset from Weights & Biases if available.
    If not found, prepare it locally and log it as an artifact.

    Parameters
    ----------
    test_size_ : float, optional (default=0.2)
        Proportion of the dataset to include in the test split.
    random_state_ : int, optional (default=42)
        Random seed for reproducibility.
    artifact_name : str, optional
        The name of the dataset artifact in W&B.

    Returns
    -------
    x_train : pd.DataFrame
    x_test : pd.DataFrame
    y_train : pd.Series
    y_test : pd.Series
    """

    x_train, x_test, y_train, y_test = prepare_train_dataset(
        test_size_=test_size_, random_state_=random_state_
    )
    return x_train, x_test, y_train, y_test

