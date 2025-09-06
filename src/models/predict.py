import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ..serving.request_models import PredictionRequest, PredictionResponse
def prediction(model, config, x_test_, y_test_):
    """
    Evaluate a trained model on test data and log results to W&B.

    Parameters
    ----------
    model : sklearn estimator
        A trained scikit-learn compatible model or pipeline.
    config : dict
        Dictionary containing experiment configuration. Must include:
            - "project_name": str, W&B project name
            - "run_name": str, descriptive name for this run
            - "params": dict, model/training hyperparameters
    x_test_ : pd.DataFrame or np.ndarray
        Test features.
    y_test_ : pd.Series or np.ndarray
        Test labels.

    Returns
    -------
    results : dict
        Dictionary with evaluation metrics: accuracy, precision, recall, f1.
    """
    y_pred = model.predict(x_test_)

    accuracy = accuracy_score(y_test_, y_pred)
    precision = precision_score(y_test_, y_pred)
    recall = recall_score(y_test_, y_pred)
    f1 = f1_score(y_test_, y_pred)
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return results