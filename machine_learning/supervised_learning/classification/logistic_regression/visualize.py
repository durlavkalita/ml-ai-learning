import numpy as np
import matplotlib.pyplot as plt
import joblib
from data import load_data

def plot_decision_boundary():
  X,y = load_data()
  model = joblib.load("model.pkl")

  plt.figure(figsize=(8, 6))
  for label in [0, 1]:
    plt.scatter(X[y == label][:, 0], X[y == label][:, 1], label=f"Class {label}", s=60)

  x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
  x2_vals = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
  xx1, xx2 = np.meshgrid(x1_vals, x2_vals)
  grid = np.c_[xx1.ravel(), xx2.ravel()]
  probs = model.predict(grid).reshape(xx1.shape)

  plt.contourf(xx1, xx2, probs, alpha=0.3, cmap="bwr")
  plt.xlabel("Hours Studied")
  plt.ylabel("Sleep Hours")
  plt.title("Decision Boundary (Sklearn)")
  plt.legend()
  plt.grid(True)
  plt.show()

if __name__ == "__main__":
  plot_decision_boundary()