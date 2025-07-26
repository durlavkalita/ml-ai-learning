import joblib
import numpy as np

def predict(inputs):
  model = joblib.load("model.pkl")
  preds = model.predict(inputs)
  return preds

if __name__ == "__main__":
  X_test = np.array([
      [3, 5],
      [9, 9],
      [6, 6]
  ])
  predictions = predict(X_test)
  print("Predictions:", predictions)