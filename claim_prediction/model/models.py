import xgboost as xgb
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def build_and_fit_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass


class XGB(BaseModel):

    def __init__(self):
        pass

    def build_and_fit_model(self, X, y, eval_set, model_params):
        self.model = xgb.XGBClassifier(
            early_stopping_rounds=15, enable_categorical=True, **model_params
        )
        self.model.fit(X, y, eval_set=eval_set, verbose=False)

    def get_predictions(self, X):
        pred = self.model.predict(X)
        pred_proba = self.model.predict_proba(X)
        return pred, pred_proba

    def load_model(self):
        """load model from model registry"""
        pass

    def save_model(self, path):
        self.model.save_model(path)

class ChampionChallenger():
    """
    Class to with method to test wether a newly trained model 
    is better than an existing model. To be run when model is retrained
    and should incorporate statistical tests (i.e t-test) to confirm
    new model is an improvement before it is registered.
    """

    def __init__(self, champion_model, challenger_model, test_data):
        pass
        
