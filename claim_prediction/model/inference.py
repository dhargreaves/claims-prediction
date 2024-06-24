

class Inference():

    def __init__(self, data_loader, preprocessor, splitter, model, writer):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.splitter = splitter
        self.model = model
        self.writer = writer

    def _load_and_preprocess_data(self):
        inference_data = self.data_loader.load_data()
        processed_data = self.preprocessor.preprocess_data(inference_data)
        self.X = self.splitter.split_data(processed_data)

    def _make_predictions(self):
        self.model.load_model()
        pred, pred_proba = self.model.get_predictions(self.X)
        self.writer.write_data(pred)

    def run_inference(self):
        self._load_and_preprocess_data()
        self._make_predictions(self)

