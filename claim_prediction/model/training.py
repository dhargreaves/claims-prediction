import mlflow
import datetime

class Trainer:

    def __init__(self, data_loader, preprocessor, parameter_tuner, model, evaluator):
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
        self.X_train, self.X_test, self.y_train, self.y_test, self.eval_set = (
            self.preprocessor.preprocess_data(training_data)
        )

    def _train_and_evaluate_model(self):
        best_params = self.parameter_tuner.tune_parameters(
            self.X_train, self.y_train, self.eval_set
        )
        self.model.build_and_fit_model(
            self.X_train, self.y_train, self.eval_set, best_params
        )
        self.train_pred, train_pred_proba = self.model.get_predictions(self.X_train)
        test_pred, test_pred_proba = self.model.get_predictions(self.X_test)
        self.evaluator.evaluate_model(
            self.y_train,
            self.y_test,
            self.train_pred,
            test_pred,
            train_pred_proba,
            test_pred_proba,
        )

    def run_training(self):

        self._load_and_process_data()
        self._train_and_evaluate_model()

        for key, value in self.evaluator.evaluation_metrics.items():
            print(key, ":", value)


class DatabricksTrainer(Trainer):
    """
    A class for training machine learning models using Databricks with MLflow integration.

    Inherits from Trainer and provides additional functionality for logging models and metrics using MLflow on Databricks.

    Attributes
    ----------
    data_loader : DataLoader
        The data loader object responsible for loading data.
    preprocessor : Preprocessor
        The preprocessor object used for preprocessing data.
    parameter_tuner : ParameterTuner
        The parameter tuner object for tuning model hyperparameters.
    model : Model
        The machine learning model object.
    evaluator : Evaluator
        The evaluator object for evaluating model performance.

    Methods
    -------
    __init__(data_loader, preprocessor, parameter_tuner, model, evaluator):
        Initializes a DatabricksTrainer instance with provided data loader, preprocessor,
        parameter tuner, model, and evaluator.

    _log_model():
        Logs the trained model using MLflow.Uses MLflow to log the trained XGBoost model with inferred signature, default Conda environment, and registers it with a specific model name.

    _train_and_evaluate_model():
        Trains the model, evaluates its performance, and logs metrics and parameters using MLflow.
        Sets the MLflow tracking URI to Databricks, starts a new experiment run with a timestamped name, trains the model, logs its parameters, and logs evaluation metrics.

    run_training():
        Runs the entire training pipeline using DatabricksTrainer.
        Disables automatic logging with MLflow, loads and processes data,
        trains and evaluates the model, and logs the trained model.
    """


    def __init__(self, data_loader, preprocessor, parameter_tuner, model, evaluator):
        super().__init__(data_loader, preprocessor, parameter_tuner, model, evaluator)

    def _log_model(self):
        signature = mlflow.models.infer_signature(self.X_train, self.train_pred)
        mlflow.xgboost.log_model(
            xgb_model=self.model.model,
            artifact_path="claim_prediction_xgb",
            conda_env=mlflow.spark.get_default_conda_env(),
            signature=signature,
            model_format="json",
            registered_model_name="claim_prediction_xgb",
        )

    def _train_and_evaluate_model(self):
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(f"/Workspace/Shared/claims_prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        with mlflow.start_run(run_name="claim_prediction"):
            super()._train_and_evaluate_model()
            mlflow.log_params(self.model.model.get_params())
            for key, value in self.evaluator.evaluation_metrics.items():
                mlflow.log_metric(key, value)

    def run_training(self):
        mlflow.autolog(disable=True)
        self._load_and_process_data()
        self._train_and_evaluate_model()
        self._log_model()
