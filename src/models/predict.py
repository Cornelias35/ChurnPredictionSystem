from ..serving.request_models import PredictionRequest, PredictionResponse
import joblib
import pandas as pd

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
    model = joblib.load(f"{model_path}.pkl")

    df = pd.DataFrame([d.model_dump() for d in prediction_request.data])

    # Make predictions
    preds = model.predict(df)

    return PredictionResponse(predictions=preds.tolist(), status="success")