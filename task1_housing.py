
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
homes = pd.read_csv("real_estate.csv")

# Convert distance into units of 100 metres
homes["distance_100m"] = (
    homes["X3 distance to the nearest MRT station"] / 100
)

# Select input and output
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Display slope and intercept
print("Task 1: Housing Price Prediction")
print("--------------------------------")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict price at 500 metres
new_distance = pd.DataFrame({
    "distance_100m": [5]
})

prediction = model.predict(new_distance)

print("Predicted price at 500 metres:", prediction[0])

# Plot observations and fitted regression line
plt.figure(figsize=(9, 6))

plt.scatter(
    X["distance_100m"],
    y,
    alpha=0.6,
    label="Actual observations"
)

plt.plot(
    X["distance_100m"],
    model.predict(X),
    color="red",
    label="Regression line"
)

plt.xlabel("Distance to MRT station (100 metres)")
plt.ylabel("House price per unit area")
plt.title("Housing Price vs Distance to MRT Station")
plt.legend()
plt.grid(True)

plt.savefig("task1_housing.png", dpi=300)
plt.show()

# Explain the slope
print("\nInterpretation:")
print(
    "For every additional 100 metres from the MRT station, "
    "the predicted house price changes by",
    round(model.coef_[0], 2),
    "price units."
)
