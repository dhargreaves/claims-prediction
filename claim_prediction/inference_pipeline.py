from model.inference import Inference
from data_processing.load_data import InferenceDataLoader
from data_processing.preprocessing import InferencePreprocesser
from output.inference_ouput import WriteToCatalog
from model.models import XGB
from config import inference_write_path
import numpy as np

if __name__ == "__main__":
    np.random.seed(1889)
    inference_pipeline = Inference(
        InferenceDataLoader(),
        InferencePreprocesser(),
        XGB(),
        WriteToCatalog(inference_write_path),
    )
    inference_pipeline.run_inference()
