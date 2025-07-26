import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# load data
df = pd.read_csv("../data/house_prices_multi.csv")
X = df[['size','bedrooms','age']]
y = df['price']

# pipeline: normalization + regression
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X,y)

# save model
joblib.dump(model, "../models/linear_regression_multi.pkl")
print("Model saved")