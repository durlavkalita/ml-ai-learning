import joblib

model = joblib.load("../models/linear_regression_multi.pkl")

input_data = [
  [1400,3,10],
  [1800,4,5],
  [1200,2,20]
]

predictions = model.predict(input_data)

for features, price in zip(input_data, predictions):
    print(f"Size: {features[0]} sqft, Bedrooms: {features[1]}, Age: {features[2]} → Predicted Price: ${price:.2f}K")