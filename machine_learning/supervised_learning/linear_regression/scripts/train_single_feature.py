import matplotlib.pyplot as plt
from utils import load_house_price_data

# load data
xs,ys = load_house_price_data("../data/house_prices.csv")
m = len(xs)

# initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.000001
epochs = 1000
tolerance = 1e-6
prev_cost = float('inf')

cost_history = []

# gradient descent loop
for epoch in range(epochs):
  dw = 0.0
  db = 0.0

  for i in range(m):
    x = xs[i]
    y = ys[i]
    y_pred = w*x+b
    error = y_pred - y
    dw += error*x
    db += error
  
  dw /= m
  db /= m

  w -= learning_rate * dw
  b -= learning_rate * db

  if epoch % 100 == 0:
    cost = sum((w*xs[i]+b-ys[i])**2 for i in range(m)) / m
    cost_history.append(cost)
    if abs(cost-prev_cost) < tolerance:
      print(f"Stopped at epoch {epoch}")
      break
    prev_cost = cost
    print(f"Epoch {epoch}: w = {w:.4f}, b = {b:.4f}, cost = {cost:.4f}")

# final model
print(f"\nTrained model: w = {w:.4f}, b = {b:.4f}")

# plot
plt.scatter(xs, ys, label="Data")
plt.plot(xs, [w * x + b for x in xs], color="red", label="Regression Line")
plt.xlabel("Size (sqft)")
plt.ylabel("Price ($1000s)")
plt.title("Linear Regression (Single Feature)")
plt.legend()
plt.grid(True)
plt.show()