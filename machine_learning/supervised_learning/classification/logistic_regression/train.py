import joblib
from sklearn.linear_model import LogisticRegression
from data import load_data

def train_and_save_model():
  X,y = load_data()

  model = LogisticRegression()
  model.fit(X,y)

  joblib.dump(model, 'model.pkl')

if __name__ == "__main__":
  train_and_save_model()