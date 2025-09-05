from src.data_prep.preprocess import load_dataset
from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import wandb
import joblib
from wandb.errors import CommError
import os

def train_model(model, param_grid, artifact_name):
    """
    Train a machine learning model with hyperparameter search and SMOTE
    oversampling, using cross-validation for evaluation.

    Parameters
    ----------
    model : estimator object
        A scikit-learn compatible estimator (e.g., RandomForestClassifier,
        LogisticRegression) to be trained.
    param_grid : dict
        Dictionary with hyperparameter search space for RandomizedSearchCV.
        Must include a key 'model_name' specifying the pipeline step name
        for the model.
        Other values should be model specific, refer to model documentation.

    Returns
    -------
    best_model : sklearn.pipeline.Pipeline
        The fitted pipeline containing the best hyperparameters and the
        trained model.
    """
    run = wandb.init(project="customer-churn", job_type="train-model")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = load_dataset(test_size_=0.25)

    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        (param_grid['model_name'], model)
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="recall",
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    model_path = f"{artifact_name}.pkl"
    artifact = wandb.Artifact(artifact_name, type="model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    run.finish()
    return best_model

def load_or_train_model(model, param_grid, artifact_name="churn-model"):
    """
    Load the latest model from Weights & Biases if available,
    otherwise train a new model and upload it.

    Parameters
    ----------
    model : estimator object
        A scikit-learn compatible estimator to be trained if no
        artifact is found.
    param_grid : dict
        Dictionary with hyperparameter search space for RandomizedSearchCV.
        Must include a key 'model_name' specifying the pipeline step name
        for the model.
    artifact_name : str, optional (default="churn-model")
        The name of the model artifact in W&B.

    Returns
    -------
    best_model : sklearn.pipeline.Pipeline
        The fitted pipeline containing the best hyperparameters and the
        trained model.
    """
    run = wandb.init(project="customer-churn", job_type="load-model", reinit=True)

    try:
        artifact = run.use_artifact(f"{artifact_name}:latest")
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, f"{artifact_name}.pkl")
        best_model = joblib.load(model_path)
        run.finish()
        return best_model

    except CommError:
        run.finish()
        return train_model(model, param_grid, artifact_name=artifact_name)


