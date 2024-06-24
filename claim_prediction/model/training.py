
class Trainer():

    def __init__(self, data_loader, preprocessor, splitter, model, evaluator):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.model = model
        self.evaluator = evaluator

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.eval_set = None

    def _load_and_process_data(self):
        training_data = self.data_loader.load_data()
        processed_data = self.preprocessor.preprocess_data(training_data)
        self.X_train, self.X_test, self.y_train, self.y_test, self.eval_set = self.splitter.split_data(processed_data)

    def _train_and_evaluate_model(self):
        self.model.fit_model(self.X_train, self.y_train, self.eval_set)
        train_pred,train_pred_proba = self.model.get_predictions(self.X_train)
        test_pred,test_pred_proba = self.model.get_predictions(self.X_test)
        self.evaluator.evaluate_model(self.y_train,self.y_test,train_pred, test_pred, train_pred_proba, test_pred_proba)

    def run_training(self):

        self._load_and_process_data()
        self._train_and_evaluate_model()

        for key, value in self.evaluator.evaluation_metrics.items():
            print(key,':',value)



