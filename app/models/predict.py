from app.serving.request_models import PredictionRequest, PredictionResponse
import joblib
import pandas as pd
from app.data_prep.preprocess import transforming_data, scale_numeric_columns

def prediction(prediction_request: PredictionRequest) -> PredictionResponse:
    """
    Evaluate a trained model on test data and log results to W&B.

    Parameters
    ----------
    :param prediction_request: PredictionRequest

    Returns
    -------
    results : dict
        Dictionary with evaluation metrics: accuracy, precision, recall, f1.
    """


    model_path = prediction_request.model_name
    model = joblib.load(f"app/paths/{model_path}.pkl")
    if model is None:
        raise Exception(f"Model {model_path} not found")
    df = pd.DataFrame([d.model_dump() for d in prediction_request.data])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna(subset=["TotalCharges"])
    df = df.drop("customerID", axis=1)

    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    service_cols = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                    "StreamingMovies"]
    multi_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]

    df = transforming_data(df, binary_cols, service_cols, multi_cols)

    num_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    df = scale_numeric_columns(x_train_=df, numeric_columns_=num_columns)
    print(df.head(1))
    print("-----------------------------")
    print(df.columns)
    print("-----------------------------")
    # Make predictions
    preds = model.predict(df)

    return PredictionResponse(predictions=preds.tolist(), status="success")