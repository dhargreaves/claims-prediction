from model.training import Trainer, DatabricksTrainer
from data_processing.load_data import TrainingDataLoader
from data_processing.preprocessing import TrainingPreprocesser
from model.models import XGB
from model.evaluation import Evaluation
from model.hyperparameter_tuning import RandomSearchTuner
from config import eval_metrics
import numpy as np

np.random.seed(1889)

if __name__ == "__main__":
    training_pipeline = DatabricksTrainer(
        TrainingDataLoader(),
        TrainingPreprocesser(),
        RandomSearchTuner(eval_metrics),
        XGB(),
        Evaluation(),
    )
    training_pipeline.run_training()
