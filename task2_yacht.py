
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score

# Load dataset
yachts = pd.read_csv(
    "yacht_hydrodynamics.data",
    sep=r"\s+",
    header=None
)

# Select Froude number and resistance
X = yachts[[5]]
y = yachts[6]

# Create the same five folds for both models
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Model 1: Linear Regression
linear_model = LinearRegression()

# Model 2: Degree-2 Polynomial Regression
quadratic_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

# Evaluate linear model
linear_scores = cross_val_score(
    linear_model,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

# Evaluate quadratic model
quadratic_scores = cross_val_score(
    quadratic_model,
    X,
    y,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

# Convert negative RMSE into positive RMSE
linear_rmse = -linear_scores
quadratic_rmse = -quadratic_scores

# Display results
print("Task 2: Yacht Resistance Model Comparison")
print("-----------------------------------------")

print("\nLinear Regression Fold RMSE:")
for i, score in enumerate(linear_rmse, start=1):
    print(f"Fold {i}: {score:.4f}")

print("Linear Mean RMSE:", linear_rmse.mean())

print("\nQuadratic Regression Fold RMSE:")
for i, score in enumerate(quadratic_rmse, start=1):
    print(f"Fold {i}: {score:.4f}")

print("Quadratic Mean RMSE:", quadratic_rmse.mean())

# Select the better model
if quadratic_rmse.mean() < linear_rmse.mean():
    print("\nBetter Model: Quadratic Regression")
else:
    print("\nBetter Model: Linear Regression")
