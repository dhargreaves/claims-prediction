from config import non_numerical_columns, features, target
from sklearn.model_selection import train_test_split


class BasePreprocessor:

    def __init__(self):
        pass

    def _clean_data(self):
        self.data.drop(columns=["family_history_3", "employment_type"], inplace=True)

    def _convert_data_types(self):
        for column in non_numerical_columns:
            self.data[column] = self.data[column].astype("category")

    def preprocess_data(self, data):
        self.data = data
        self._clean_data()
        self._convert_data_types()
        return self.data


class TrainingPreprocesser(BasePreprocessor):

    def __init__(self):
        super().__init__()

    def _select_features(self):
        self.X, self.y = self.data[features], self.data[target]

    def _split_data(self):
        # split into train and test
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=1889
        )
        # split train into train and validation sets
        self.X_train, X_eval, self.y_train, y_eval = train_test_split(
            X_train, y_train, test_size=0.2, random_state=101
        )
        self.eval_set = [(X_eval, y_eval)]

    def preprocess_data(self, data):
        self.data = data
        self._clean_data()
        self._convert_data_types()
        self._select_features()
        self._split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test, self.eval_set


class InferencePreprocesser(BasePreprocessor):

    def __init__(self):
        super().__init__()

    def _select_features(self):
        self.X = self.data[features]

    def preprocess_data(self, data):
        self.data = data
        self._clean_data()
        self._convert_data_types()
        self._select_features()
        return self.X
