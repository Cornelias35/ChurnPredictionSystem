from app.data_prep.preprocess import load_dataset
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib
from app.serving.request_models import AVAILABLE_MODELS, TrainingRequest, ModelResponse
from app.models.evaluate import get_metrics

def train_model(train_request : TrainingRequest):
    """
    Train a machine learning model with hyperparameter search and SMOTE
    oversampling, using cross-validation for evaluation.

    Parameters
    ----------
    :param train_request: TrainingRequest

    Returns
    -------
    best_model : sklearn.pipeline.Pipeline
        The fitted pipeline containing the best hyperparameters and the
        trained model.

    """
    cv = StratifiedKFold(n_splits=train_request.cv_folds, shuffle=True, random_state=42)

    X_train, X_test, y_train, y_test = load_dataset(test_size_=train_request.test_size)
    model_class = AVAILABLE_MODELS[train_request.model_name]
    model = model_class()

    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        (train_request.model_name, model)
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=train_request.parameters,
        n_iter=train_request.n_iter,
        scoring=train_request.best_metric,
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    if hasattr(best_model, "predict_proba"):
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        y_pred_proba = best_model.decision_function(X_test)
    else:
        y_pred_proba = None
    results = get_metrics(y_test, y_pred, y_pred_proba)

    model_path = f"app/paths/{train_request.model_name}.pkl"
    joblib.dump(best_model, model_path)

    return ModelResponse(
        model_name=train_request.model_name,
        best_score=results[train_request.best_metric],
        metrics=results,
        status="Completed"

    )


