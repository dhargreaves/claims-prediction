from claim_prediction.model.training import Train
from data.load_data import TrainingDataLoader
from data.preprocessing import Preprocessing, TrainingSplitter
from model.models import XGB
from model.evaluation import Evaluation
import numpy as np

if __name__ == '__main__':
    np.random.seed(1889)
    training_pipeline = Train(TrainingDataLoader(),
                  Preprocessing(),
                  TrainingSplitter(test_size=0.2),
                  XGB(),
                  Evaluation())
    training_pipeline.run_training()