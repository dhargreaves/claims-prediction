from config import non_numerical_columns, features, target
from sklearn.model_selection import train_test_split


class BasePreprocessor:
    """
    Base class for data preprocessing.

    Methods
    -------
    __init__():
        Initializes an instance of the BasePreprocessor class.

    _clean_data():
        Cleans the data by dropping specified columns.

    _convert_data_types():
        Converts non-numeric columns to categorical data types.

    preprocess_data(data):
        Performs data preprocessing steps on the input data.

        Parameters
        ----------
        data : pandas.DataFrame
            The input DataFrame to be preprocessed.
    """
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


class TrainingPreprocesser(BasePreprocessor):
    """
        Performs data preprocessing steps specific to training data.

        Parameters
        ----------
        data : pandas.DataFrame
            The input DataFrame to be preprocessed.

        Returns
        -------
        pandas.DataFrame
            The training set (X_train).
        pandas.DataFrame
            The test set (X_test).
        pandas.DataFrame
            The training labels (y_train).
        pandas.DataFrame
            The test labels (y_test).
        list of tuple
            The evaluation set, where each tuple contains (X_eval, y_eval).
    """   

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
        super().preprocess_data(data)
        self._select_features()
        self._split_data()
        return self.X_train, self.X_test, self.y_train, self.y_test, self.eval_set


class InferencePreprocesser(BasePreprocessor):
    """
    Subclass of BasePreprocessor specialized for inference data preprocessing.

    Methods
    -------
    __init__():
        Initializes an instance of the InferencePreprocessor class.

    _select_features():
        Selects features (X) columns from self.data.

    preprocess_data(data):
        Performs data preprocessing steps specific to inference data.

        Parameters
        ----------
        data : pandas.DataFrame
            The input DataFrame to be preprocessed.

        Returns
        -------
        pandas.DataFrame
            The preprocessed features (X).
    """

    def __init__(self):
        super().__init__()

    def _select_features(self):
        self.X = self.data[features]

    def preprocess_data(self, data):
        super().preprocess_data(data)
        self._select_features()
        return self.X
