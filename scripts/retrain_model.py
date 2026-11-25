# 1. Import required libraries
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.pyfunc

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 2. Configure project directory and MLflow
BASE_DIR = Path(__file__).resolve().parent.parent

TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

# 3. Configure registered model
MODEL_NAME = "HousePriceDriftModel"

# 4. Load existing champion model
champion_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

print("Champion model loaded successfully!")

# 5. Load recent labelled data
df = pd.read_csv(
    BASE_DIR / "data" / "retraining_data.csv"
)

# 6. Define input features and target
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

X = df[FEATURES]
y = df["price"]

print("\nRetraining dataset shape:", df.shape)

# 7. Split recent data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 8. Initialize challenger model
challenger_model = LinearRegression()

# 9. Train challenger on recent market data
challenger_model.fit(
    X_train,
    y_train
)

print("\nChallenger model trained successfully!")

# 10. Generate champion predictions
champion_predictions = champion_model.predict(
    X_test
)

# 11. Generate challenger predictions
challenger_predictions = challenger_model.predict(
    X_test
)


# 12. Create reusable model evaluation function
def evaluate_model(model_name, actual, predictions):

    # 12.1 Calculate MAE
    mae = mean_absolute_error(
        actual,
        predictions
    )

    # 12.2 Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    # 12.3 Calculate R2 score
    r2 = r2_score(
        actual,
        predictions
    )

    # 12.4 Return evaluation metrics
    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# 13. Evaluate champion model
champion_results = evaluate_model(
    "champion",
    y_test,
    champion_predictions
)

# 14. Evaluate challenger model
challenger_results = evaluate_model(
    "challenger",
    y_test,
    challenger_predictions
)

# 15. Create model comparison DataFrame
results_df = pd.DataFrame([
    champion_results,
    challenger_results
])

# 16. Display model comparison
print("\nChampion vs Challenger Results:")

print(
    results_df.round({
        "mae": 2,
        "rmse": 2,
        "r2": 6
    }).to_string(index=False)
)

# 17. Calculate percentage improvement in MAE
mae_improvement = (
    (
        champion_results["mae"]
        - challenger_results["mae"]
    )
    / champion_results["mae"]
) * 100

print(
    f"\nMAE Improvement: {mae_improvement:.2f}%"
)

# 18. Display challenger model coefficients
print("\nChallenger Model Coefficients:")

for feature, coefficient in zip(
    FEATURES,
    challenger_model.coef_
):
    print(f"{feature}: {coefficient:.2f}")

print(
    f"Intercept: {challenger_model.intercept_:.2f}"
)

# 19. Load historical reference dataset
reference = pd.read_csv(
    BASE_DIR / "data" /
    "monitoring_reference.csv"
)

X_reference = reference[FEATURES]
y_reference = reference["price"]

# 20. Evaluate champion on historical data
historical_champion_predictions = (
    champion_model.predict(X_reference)
)

historical_champion_results = evaluate_model(
    "historical_champion",
    y_reference,
    historical_champion_predictions
)

# 21. Evaluate challenger on historical data
historical_challenger_predictions = (
    challenger_model.predict(X_reference)
)

historical_challenger_results = evaluate_model(
    "historical_challenger",
    y_reference,
    historical_challenger_predictions
)

# 22. Create historical comparison report
historical_results_df = pd.DataFrame([
    historical_champion_results,
    historical_challenger_results
])

print("\nHistorical Data Evaluation:")

print(
    historical_results_df.round({
        "mae": 2,
        "rmse": 2,
        "r2": 6
    }).to_string(index=False)
)

# 23. Define model acceptance thresholds
MIN_MAE_IMPROVEMENT = 20
MAX_CHALLENGER_MAE = 150000
MIN_CHALLENGER_R2 = 0.95

# 24. Evaluate individual acceptance criteria
improvement_passed = (
    mae_improvement >= MIN_MAE_IMPROVEMENT
)

mae_passed = (
    challenger_results["mae"]
    <= MAX_CHALLENGER_MAE
)

r2_passed = (
    challenger_results["r2"]
    >= MIN_CHALLENGER_R2
)

# 25. Determine challenger acceptance
candidate_accepted = (
    improvement_passed
    and mae_passed
    and r2_passed
)

# 26. Display individual acceptance results
print("\nModel Acceptance Gate:")

print("MAE improvement passed:", improvement_passed)
print("MAE threshold passed:", mae_passed)
print("R2 threshold passed:", r2_passed)

# 27. Display final acceptance decision
if candidate_accepted:

    print("\nCandidate accepted for registration.")

else:

    print("\nCandidate rejected.")
    print("Further investigation required.")

# 28. Save model comparison report
results_df.to_csv(
    BASE_DIR / "reports" /
    "retraining_comparison.csv",
    index=False
)

# 29. Save historical comparison report
historical_results_df.to_csv(
    BASE_DIR / "reports" /
    "historical_model_comparison.csv",
    index=False
)

# 30. Save acceptance gate results
acceptance_df = pd.DataFrame([{
    "mae_improvement_pct": mae_improvement,
    "improvement_passed": bool(improvement_passed),
    "mae_passed": bool(mae_passed),
    "r2_passed": bool(r2_passed),
    "candidate_accepted": bool(candidate_accepted)
}])

acceptance_df.to_csv(
    BASE_DIR / "reports" /
    "model_acceptance_report.csv",
    index=False
)

# 31. Save challenger model only if accepted
if candidate_accepted:

    model_path = (
        BASE_DIR / "models" /
        "challenger_model.joblib"
    )

    joblib.dump(
        challenger_model,
        model_path
    )

    print("\nChallenger model saved:", model_path)

# 32. Display completion message
print("\nModel retraining and evaluation completed!")