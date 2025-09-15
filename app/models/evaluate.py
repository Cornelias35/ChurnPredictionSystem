from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_metrics(y_test_, y_pred, y_pred_proba = None):
    accuracy = accuracy_score(y_test_, y_pred)
    precision = precision_score(y_test_, y_pred, zero_division=0)
    recall = recall_score(y_test_, y_pred, zero_division=0)
    f1 = f1_score(y_test_, y_pred, zero_division=0)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if y_pred_proba is not None and len(set(y_test_)) > 1:
        roc_auc = roc_auc_score(y_test_, y_pred_proba)
        metrics["roc_auc"] = roc_auc

    return metrics