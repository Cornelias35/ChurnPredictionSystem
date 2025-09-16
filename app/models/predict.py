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

    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({"Yes": 1, "No": 0}))

    df[service_cols] = df[service_cols].apply(
        lambda x: x.map({"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}))

    df["gender_Female"] = (df["gender"] == "Female").astype(int)
    df["gender_Male"] = (df["gender"] == "Male").astype(int)

    df["InternetService_DSL"] = (df["InternetService"] == "DSL").astype(int)
    df["InternetService_Fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["InternetService_No"] = (df["InternetService"] == "No").astype(int)

    df["Contract_Month-to-month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["Contract_One_year"] = (df["Contract"] == "One year").astype(int)
    df["Contract_Two_year"] = (df["Contract"] == "Two year").astype(int)

    df["PaymentMethod_Bank_transfer_automatic"] = (df["PaymentMethod"] == "Bank transfer (automatic)").astype(int)
    df["PaymentMethod_Credit_card_automatic"] = (df["PaymentMethod"] == "Credit card (automatic)").astype(int)
    df["PaymentMethod_Electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["PaymentMethod_Mailed_check"] = (df["PaymentMethod"] == "Mailed check").astype(int)

    df = df.drop(["Contract", "InternetService", "PaymentMethod", "gender"], axis=1)

    num_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

    df = scale_numeric_columns(x_train_=df, numeric_columns_=num_columns)
    preds = model.predict(df)

    return PredictionResponse(predictions=preds.tolist(), status="success")