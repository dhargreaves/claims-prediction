from model.inference import Inference
from data.load_data import InferenceDataLoader
from data.preprocessing import Preprocessing, InferenceSplitter
from output.inference_ouput import WriteToCatalog
from model.models import XGB
from config import inference_write_path
import numpy as np

if __name__ == '__main__':
    np.random.seed(1889)
    inference_pipeline = Inference(InferenceDataLoader(),
                                    Preprocessing(),
                                    InferenceSplitter(),
                                    XGB(),
                                    WriteToCatalog(inference_write_path))
    inference_pipeline.run_inference()