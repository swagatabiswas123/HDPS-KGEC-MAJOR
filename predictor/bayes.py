from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd

class GaussianNaiveBayesModel():
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = GaussianNB()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None
    

def find():
    file_path = "C:\\Users\\swagato\\\\Downloads\\HDPS-FocusUs-master\\predictor\\heartdata.csv"
    return load_data(file_path)