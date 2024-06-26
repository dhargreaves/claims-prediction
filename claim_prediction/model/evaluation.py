import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    roc_curve,
)


class Evaluation:

    def __init__(self):
        self.evaluation_metrics = {}

    def _get_cohen_kappa_score(self, y, train_pred):
        y = np.array(y)
        y = y.astype(int)
        yhat = np.array(train_pred)
        yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
        kappa = round(cohen_kappa_score(yhat, y, weights="quadratic"), 2)
        return kappa

    def evaluate_model(
        self, y_train, y_test, train_pred, test_pred, train_pred_proba, test_pred_proba
    ):
        self.evaluation_metrics["train_kappa_score"] = self._get_cohen_kappa_score(
            y_train, train_pred
        )
        self.evaluation_metrics["test_kappa_score"] = self._get_cohen_kappa_score(
            y_test, test_pred
        )
        self.evaluation_metrics["train_accuracy"] = accuracy_score(y_train, train_pred)
        self.evaluation_metrics["test_accuracy"] = accuracy_score(y_test, test_pred)
        self.train_confusion_matrix = confusion_matrix(y_train, train_pred)
        self.test_confusion_matrix = confusion_matrix(y_test, test_pred)
        self.evaluation_metrics["train_roc"] = roc_auc_score(y_train, train_pred)
        self.evaluation_metrics["test_roc"] = roc_auc_score(y_test, test_pred)
        self.evaluation_metrics["train_log_loss"] = log_loss(y_train, train_pred_proba)
        self.evaluation_metrics["test_log_loss"] = log_loss(y_test, test_pred_proba)
        self.evaluation_metrics["f1_score"] = f1_score(y_test, test_pred)
        self.evaluation_metrics["precision"] = precision_score(y_test, test_pred)
        self.evaluation_metrics["recall"] = recall_score(y_test, test_pred)
