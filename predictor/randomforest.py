from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

class RandomForestModel():
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
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
