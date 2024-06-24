from model.training import Trainer, DatabricksTrainer
from data_processing.load_data import TrainingDataLoader
from data_processing.preprocessing import Preprocessing, TrainingSplitter
from model.models import XGB
from model.evaluation import Evaluation
import numpy as np

np.random.seed(1889)

if __name__ == '__main__':
    training_pipeline = DatabricksTrainer(TrainingDataLoader(),
                  Preprocessing(),
                  TrainingSplitter(test_size=0.2),
                  XGB(),
                  Evaluation())
    training_pipeline.run_training()