from config import non_numerical_columns, features, target
from sklearn.model_selection import train_test_split
from abc import ABC,  abstractmethod

class Preprocessing():

    def __init__(self):
        pass

    def _clean_data(self):
        self.data.drop(columns=['family_history_3', 'employment_type'], inplace=True)

    def _convert_data_types(self):
        for column in non_numerical_columns:
            self.data[column] = self.data[column].astype('category')

    def preprocess_data(self, data):
        self.data = data
        self._clean_data()
        self._convert_data_types()
        return self.data

class Splitter(ABC):

    @abstractmethod
    def split_data(self):
        pass

class TrainingSplitter(Splitter):

    def __init__(self,test_size):
        self.test_size = test_size

    def split_data(self, data):
        X, y = data[features], data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1889)
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=self.test_size, random_state=101)
        eval_set = [(X_eval, y_eval)]
        return X_train, X_test, y_train, y_test, eval_set
    
class InferenceSplitter(Splitter):

    def __init__(self):
        pass

    def split_data(self,data):
        return data[features]
        

        