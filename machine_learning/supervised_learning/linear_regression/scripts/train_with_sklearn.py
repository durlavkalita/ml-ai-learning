from sklearn.linear_model import LinearRegression
import joblib
from utils import load_house_price_data

# load data
X,y = load_house_price_data("../data/house_prices.csv", as_dataframe=True)

# train model
model = LinearRegression()
model.fit(X,y)

# save model
joblib.dump(model, "../models/linear_regression_model.pkl")
print("Model saved.")
