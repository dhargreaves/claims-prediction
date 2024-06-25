#import mlflow

class Trainer():

    def __init__(self, 
                 data_loader, 
                 preprocessor, 
                 parameter_tuner, 
                 model, evaluator):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.parameter_tuner = parameter_tuner
        self.model = model
        self.evaluator = evaluator

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.eval_set = None

    def _load_and_process_data(self):
        training_data = self.data_loader.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test, self.eval_set = self.preprocessor.preprocess_data(training_data)

    def _train_and_evaluate_model(self):
        best_params = self.parameter_tuner.tune_parameters(self.X_train, 
                                                           self.y_train, 
                                                           self.eval_set)
        self.model.build_and_fit_model(self.X_train, 
                                        self.y_train, 
                                        self.eval_set,
                                        best_params)
        self.train_pred,train_pred_proba = self.model.get_predictions(self.X_train)
        test_pred,test_pred_proba = self.model.get_predictions(self.X_test)
        self.evaluator.evaluate_model(self.y_train,
                                      self.y_test,
                                      self.train_pred,
                                      test_pred, 
                                      train_pred_proba, 
                                      test_pred_proba)

    def run_training(self):

        self._load_and_process_data()
        self._train_and_evaluate_model()

        for key, value in self.evaluator.evaluation_metrics.items():
            print(key,':',value)

class DatabricksTrainer(Trainer):

    def __init__(self, data_loader, preprocessor, splitter, model, evaluator):
        super().__init__(data_loader, preprocessor, splitter, model, evaluator)

    def _log_model(self):
        signature = mlflow.models.infer_signature(self.X_train,self.train_pred)
        mlflow.xgboost.log_model(xgb_model=self.model.model,
                                 artifact_path='claim_prediction_xgb',
                                 conda_env=mlflow.spark.get_default_conda_env(),
                                 signature=signature,
                                 model_format='json',
                                 registered_model_name='claim_prediction_xgb')

    def _train_and_evaluate_model(self):
        with mlflow.start_run(run_name='claim_prediction'):
            super()._train_and_evaluate_model()
            mlflow.log_params(self.model.model_params)
            for key, value in self.evaluator.evaluation_metrics.items():
                mlflow.log_metric(key, value)

        
    def run_training(self):
        mlflow.autolog(disable=True)
        self._load_and_process_data()
        self._train_and_evaluate_model()
        self._log_model()



