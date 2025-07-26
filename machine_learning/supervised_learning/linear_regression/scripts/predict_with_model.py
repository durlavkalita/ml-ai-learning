import joblib
import numpy as np

model = joblib.load("../models/linear_Regression_model.pkl")

size = 1750
predicted_price = model.predict(np.array([[size]]))
print(f"Predicted price for {size}: {predicted_price[0]:.2f}")

sizes = [1200, 1500, 1800]
X = [[s] for s in sizes]

predicted_prices = model.predict(X)

for size, price in zip(sizes, predicted_prices):
    print(f"Size: {size} sqft → Predicted Price: ${price:.2f}K")