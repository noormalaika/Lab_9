
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Load dataset and remove duplicates
concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()

target = "Concrete compressive strength"

X = concrete.drop(columns=target)
y = concrete[target]

print("Dataset after removing duplicates:", concrete.shape)


# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 3. Five-fold cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 4. Linear Regression baseline
baseline_model = LinearRegression()

baseline_scores = cross_val_score(
    baseline_model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

baseline_rmse = -baseline_scores

print("\nBaseline Mean CV RMSE:", baseline_rmse.mean())


# 5. Create polynomial + scaling + Ridge pipeline
pipeline = Pipeline([
    ("poly", PolynomialFeatures()),
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])


# 6. Define hyperparameter grid
param_grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}


# 7. Tune using GridSearchCV
search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    refit=True
)

search.fit(X_train, y_train)

print("\nBest Parameters:", search.best_params_)
print("Tuned Mean CV RMSE:", -search.best_score_)


# 8. Display top three settings
results = pd.DataFrame(search.cv_results_)

top_three = results[
    [
        "rank_test_score",
        "param_poly__degree",
        "param_ridge__alpha",
        "mean_test_score"
    ]
].sort_values("rank_test_score").head(3)

top_three["mean_CV_RMSE"] = -top_three["mean_test_score"]

print("\nTop Three Hyperparameter Settings:")
print(top_three.to_string(index=False))


# 9. Evaluate the best model on test data
y_pred = search.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)

print("\nTest RMSE:", test_rmse)
print("Test MAE:", test_mae)


# 10. Actual vs predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6,
    label="Predictions"
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    label="Perfect prediction (y=x)"
)

plt.xlabel("Actual Concrete Strength (MPa)")
plt.ylabel("Predicted Concrete Strength (MPa)")
plt.title("Actual vs Predicted Concrete Strength")
plt.legend()
plt.grid(True)

plt.savefig("task3_concrete.png", dpi=300)
plt.show()


# 11. Compare baseline and tuned models
print("\nModel Comparison")
print("----------------")
print("Baseline CV RMSE:", baseline_rmse.mean())
print("Tuned CV RMSE:", -search.best_score_)
print("Improvement:", baseline_rmse.mean() - (-search.best_score_))
