import xgboost as xgb
from config import xgb_params
from abc import ABC,  abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(self):
        self._build_model()

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod 
    def fit_model(self):
        pass
    
    @abstractmethod
    def save_model(self):
        pass


class XGB(BaseModel):

    def __init__(self):
        pass

    def _build_model(self):
        self.model = xgb.XGBClassifier(early_stopping_rounds=15,
                          enable_categorical=True,
                          **xgb_params)
        
    def fit_model(self, X, y, eval_set):
        self._build_model()
        self.model.fit(X,y,eval_set=eval_set, verbose=False)

    def get_predictions(self,X):
        pred = self.model.predict(X)
        pred_proba = self.model.predict_proba(X)
        return pred, pred_proba
    
    def load_model(self):
        pass
    
    def save_model(self):
        pass

