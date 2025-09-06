from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

AVAILABLE_MODELS = {
    "knn": KNeighborsClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "xgb": XGBClassifier,
    "catboost": CatBoostClassifier,
    "svc": SVC,
    "gbdt": GradientBoostingClassifier,
    "naive_bayes": GaussianNB,
}

AVAILABLE_METRICS = {
    "accuracy":accuracy_score,
    "precision":precision_score,
    "recall":recall_score,
    "f1":f1_score
}

class PredictionData(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Credit_card_automatic: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

class TrainingRequest(BaseModel):
    model_name: str
    parameters: Dict[str, Any] = {}
    experiment_name: str = "customer-churn"
    best_metric: str = "f1"
    test_size: float = 0.25
    cv_folds: int = 5
    n_iter: int = 20

class PredictionRequest(BaseModel):
    model_name: str
    experiment_name: str = "customer-churn"
    data: List[PredictionData]

class ModelResponse(BaseModel):
    model_name: str
    best_score: float
    metrics: Dict[str, float]
    status: str

class PredictionResponse(BaseModel):
    predictions: List[int]
    status: str


