from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import xgboost as xgb


class RandomSearchTuner:

    def __init__(self, eval_metrics):
        estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=eval_metrics,
            early_stopping_rounds=15,
            enable_categorical=True,
        )
        param_distributions = {
            "n_estimators": stats.randint(50, 500),
            "learning_rate": stats.uniform(0.01, 0.75),
            "subsample": stats.uniform(0.25, 0.75),
            "max_depth": stats.randint(1, 8),
            "colsample_bytree": stats.uniform(0.1, 0.75),
            "min_child_weight": [1, 3, 5, 7, 9],
        }
        self.parameter_gridSearch = RandomizedSearchCV(
            estimator,
            param_distributions,
            cv=5,
            n_iter=100,
            verbose=False,
            random_state=101,
            scoring="roc_auc",
        )

    def tune_parameters(self, X_train, y_train, eval_set):
        self.parameter_gridSearch.fit(
            X_train, y_train, eval_set=eval_set, verbose=False
        )

        return self.parameter_gridSearch.best_params_
