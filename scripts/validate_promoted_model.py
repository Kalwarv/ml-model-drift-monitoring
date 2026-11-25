# 1. Import required libraries
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

from pathlib import Path
from mlflow import MlflowClient

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

EXPERIMENT_NAME = "HousePriceDriftMonitoring"
MODEL_NAME = "HousePriceDriftModel"

mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

# 3. Identify current and previous champion versions
current_version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
).version

previous_version = client.get_model_version_by_alias(
    MODEL_NAME,
    "previous_champion"
).version

print("Current champion:", current_version)
print("Previous champion:", previous_version)

# 4. Load both registered models
current_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{current_version}"
)

previous_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{previous_version}"
)

# 5. Load independent production concept drift data
df = pd.read_csv(
    BASE_DIR / "data" /
    "production_concept_drift.csv"
)

# 6. Define model inputs and actual target
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

X = df[FEATURES]
y = df["price"]

print("\nValidation samples:", len(df))

# 7. Validate required input values
if X.isnull().any().any() or y.isnull().any():
    raise ValueError(
        "Missing values detected in validation data."
    )

# 8. Generate predictions using both versions
previous_predictions = previous_model.predict(X)

current_predictions = current_model.predict(X)


# 9. Create reusable evaluation function
def evaluate_model(version, actual, predictions):

    # 9.1 Calculate residuals
    errors = actual - predictions

    # 9.2 Calculate evaluation metrics
    return {
        "model_version": version,
        "samples": len(actual),
        "mae": mean_absolute_error(
            actual, predictions
        ),
        "rmse": np.sqrt(
            mean_squared_error(actual, predictions)
        ),
        "r2": r2_score(
            actual, predictions
        ),
        "mean_error": errors.mean()
    }


# 10. Evaluate previous champion
previous_results = evaluate_model(
    previous_version,
    y,
    previous_predictions
)

# 11. Evaluate current champion
current_results = evaluate_model(
    current_version,
    y,
    current_predictions
)

# 12. Create model comparison report
results_df = pd.DataFrame([
    previous_results,
    current_results
])

print("\nPost-Promotion Validation Results:")

print(
    results_df.round({
        "mae": 2,
        "rmse": 2,
        "r2": 6,
        "mean_error": 2
    }).to_string(index=False)
)

# 13. Calculate MAE improvement
mae_improvement = (
    (
        previous_results["mae"]
        - current_results["mae"]
    )
    / previous_results["mae"]
) * 100

print(
    f"\nMAE Improvement: {mae_improvement:.2f}%"
)

# 14. Define recovery acceptance criteria
MIN_MAE_IMPROVEMENT = 20
MAX_MAE = 150000
MIN_R2 = 0.95

# 15. Determine whether model recovered
recovery_successful = bool(
    mae_improvement >= MIN_MAE_IMPROVEMENT
    and current_results["mae"] <= MAX_MAE
    and current_results["r2"] >= MIN_R2
)

print("\nModel Recovery Successful:",
      recovery_successful)

# 16. Store individual model predictions
df["previous_prediction"] = previous_predictions
df["current_prediction"] = current_predictions

df["previous_error"] = (
    df["price"] - df["previous_prediction"]
)

df["current_error"] = (
    df["price"] - df["current_prediction"]
)

# 17. Save prediction comparison
prediction_path = (
    BASE_DIR / "data" /
    "post_promotion_predictions.csv"
)

df.to_csv(prediction_path, index=False)

# 18. Save performance comparison
comparison_path = (
    BASE_DIR / "reports" /
    "post_promotion_validation.csv"
)

results_df.to_csv(
    comparison_path,
    index=False
)

# 19. Save model recovery decision
recovery_df = pd.DataFrame([{
    "previous_version": previous_version,
    "current_version": current_version,
    "mae_improvement_pct": mae_improvement,
    "recovery_successful": recovery_successful
}])

recovery_path = (
    BASE_DIR / "reports" /
    "model_recovery_report.csv"
)

recovery_df.to_csv(
    recovery_path,
    index=False
)

# 20. Track post-promotion validation in MLflow
with mlflow.start_run(
    run_name="post_promotion_validation"
) as run:

    # 20.1 Log validation parameters
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "previous_version": previous_version,
        "current_version": current_version,
        "validation_samples": len(df),
        "validation_dataset":
            "production_concept_drift.csv"
    })

    # 20.2 Log previous model metrics
    mlflow.log_metrics({
        "previous_mae": float(
            previous_results["mae"]
        ),
        "previous_rmse": float(
            previous_results["rmse"]
        ),
        "previous_r2": float(
            previous_results["r2"]
        )
    })

    # 20.3 Log current model metrics
    mlflow.log_metrics({
        "current_mae": float(
            current_results["mae"]
        ),
        "current_rmse": float(
            current_results["rmse"]
        ),
        "current_r2": float(
            current_results["r2"]
        ),
        "mae_improvement_pct": float(
            mae_improvement
        )
    })

    # 20.4 Add validation tags
    mlflow.set_tags({
        "training_stage": "post_promotion_validation",
        "drift_scenario": "concept_drift",
        "recovery_successful": str(
            recovery_successful
        ).lower()
    })

    # 20.5 Log validation artifacts
    for path in [
        prediction_path,
        comparison_path,
        recovery_path
    ]:

        mlflow.log_artifact(
            str(path),
            artifact_path="validation_reports"
        )

    print("\nMLflow Run ID:", run.info.run_id)

# 21. Display final validation decision
if recovery_successful:

    print("\nModel recovery verified successfully!")
    print("Current champion meets acceptance criteria.")

else:

    print("\nModel recovery verification failed.")
    print("Investigate and consider rollback.")

# 22. Display completion message
print("\nPost-promotion validation completed!")