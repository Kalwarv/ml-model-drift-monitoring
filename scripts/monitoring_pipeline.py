# 1. Import required libraries
import argparse
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

from pathlib import Path
from datetime import datetime, timezone
from mlflow import MlflowClient

from scipy.stats import (
    ks_2samp,
    chi2_contingency
)

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

# 3. Configure monitoring thresholds
DRIFT_THRESHOLD = 0.05
MAE_THRESHOLD = 0.25
R2_THRESHOLD = 0.10

# 4. Define expected model features
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

# 5. Configure command-line arguments
parser = argparse.ArgumentParser(
    description="ML model monitoring pipeline"
)

parser.add_argument(
    "--dataset",
    required=True,
    help="Production CSV filename"
)

args = parser.parse_args()

dataset_name = args.dataset

# 6. Identify current champion version
champion_version = str(
    client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    ).version
)

print("Current champion:", champion_version)

# 7. Load current champion model
model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{champion_version}"
)

# 8. Select version-specific monitoring files
if champion_version == "1":

    reference_path = (
        BASE_DIR / "data" /
        "monitoring_reference.csv"
    )

    baseline_path = (
        BASE_DIR / "reports" /
        "baseline_metrics.json"
    )

else:

    reference_path = (
        BASE_DIR / "data" /
        f"monitoring_reference_v{champion_version}.csv"
    )

    baseline_path = (
        BASE_DIR / "reports" /
        f"baseline_metrics_v{champion_version}.json"
    )

# 9. Verify monitoring reference availability
if not reference_path.exists():
    raise FileNotFoundError(
        f"Monitoring reference not found: {reference_path}"
    )

if not baseline_path.exists():
    raise FileNotFoundError(
        f"Baseline metrics not found: {baseline_path}"
    )

# 10. Load historical reference data
reference = pd.read_csv(reference_path)

# 11. Load version-specific baseline metrics
with open(baseline_path) as file:
    baseline = json.load(file)

baseline_mae = baseline["mae"]
baseline_r2 = baseline["r2"]

if baseline_mae <= 0:
    raise ValueError("Baseline MAE must be positive.")

# 12. Load incoming production data
current = pd.read_csv(
    BASE_DIR / "data" / dataset_name
)

print("\nProduction dataset:", dataset_name)
print("Production samples:", len(current))
print("Reference samples:", len(reference))

# 13. Validate production data
required_columns = FEATURES + ["price"]

missing_columns = [
    column
    for column in required_columns
    if column not in current.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if len(current) < 2:
    raise ValueError(
        "At least two records are required."
    )

if current[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values detected in production data."
    )

if not np.isfinite(
    current[required_columns].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Invalid numerical values detected."
    )

# 14. Separate inputs and actual target
X = current[FEATURES]
y = current["price"]

# 15. Generate current champion predictions
predictions = model.predict(X)

current["prediction"] = predictions

# 16. Calculate prediction residuals
current["error"] = (
    current["price"] - current["prediction"]
)

current["absolute_error"] = (
    current["error"].abs()
)

reference_errors = (
    reference["price"] - reference["prediction"]
)

# 17. Calculate current model performance
current_mae = mean_absolute_error(
    y,
    predictions
)

current_rmse = np.sqrt(
    mean_squared_error(y, predictions)
)

current_r2 = r2_score(
    y,
    predictions
)

# 18. Calculate performance degradation
mae_increase = (
    current_mae - baseline_mae
) / baseline_mae

r2_decrease = (
    baseline_r2 - current_r2
)


# 19. Create reusable Data Drift function
def detect_feature_drift(
    reference_values,
    current_values,
    feature
):

    # 19.1 Use Chi-square for bedroom counts
    if feature == "bedrooms":

        frequency_table = pd.crosstab(
            pd.Series(
                ["reference"] * len(reference_values)
                + ["current"] * len(current_values)
            ),
            pd.concat(
                [
                    reference_values,
                    current_values
                ],
                ignore_index=True
            )
        )

        statistic, p_value, _, _ = (
            chi2_contingency(frequency_table)
        )

        test_name = "Chi-square"

    # 19.2 Use KS for numerical features
    else:

        statistic, p_value = ks_2samp(
            reference_values,
            current_values
        )

        test_name = "KS"

    # 19.3 Return feature monitoring result
    return {
        "feature": feature,
        "test": test_name,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "drift_detected": bool(
            p_value < DRIFT_THRESHOLD
        )
    }


# 20. Detect Data Drift for each feature
feature_drift_results = []

for feature in FEATURES:

    result = detect_feature_drift(
        reference[feature],
        current[feature],
        feature
    )

    feature_drift_results.append(result)

# 21. Calculate drifted feature count
drifted_features = [
    result["feature"]
    for result in feature_drift_results
    if result["drift_detected"]
]

data_drift = bool(drifted_features)

# 22. Detect Prediction Drift
prediction_ks, prediction_p_value = ks_2samp(
    reference["prediction"],
    current["prediction"]
)

prediction_drift = bool(
    prediction_p_value < DRIFT_THRESHOLD
)

# 23. Detect residual distribution drift
residual_ks, residual_p_value = ks_2samp(
    reference_errors,
    current["error"]
)

residual_drift = bool(
    residual_p_value < DRIFT_THRESHOLD
)

# 24. Evaluate performance degradation
mae_degraded = bool(
    mae_increase > MAE_THRESHOLD
)

r2_degraded = bool(
    r2_decrease > R2_THRESHOLD
)

# 25. Detect potential Concept Drift
concept_drift_suspected = (
    mae_degraded
    and (residual_drift or r2_degraded)
)

# 26. Implement monitoring decision engine
if concept_drift_suspected:

    decision = "RETRAINING_REQUIRED"

    reason = (
        "Significant performance degradation "
        "with evidence of residual or R2 change."
    )

elif mae_degraded or r2_degraded:

    decision = "INVESTIGATE_PERFORMANCE"

    reason = (
        "Performance degradation requires investigation."
    )

elif data_drift or prediction_drift or residual_drift:

    decision = "MONITORING_ALERT"

    reason = (
        "Distribution changes detected without "
        "major performance degradation."
    )

else:

    decision = "CONTINUE_MONITORING"

    reason = (
        "Model performance is within monitoring thresholds."
    )

# 27. Determine retraining trigger
retraining_required = (
    decision == "RETRAINING_REQUIRED"
)

# 28. Generate unique monitoring timestamp
timestamp = datetime.now(timezone.utc).strftime(
    "%Y%m%d_%H%M%S_%f"
)

dataset_stem = Path(dataset_name).stem

report_id = (
    f"v{champion_version}_"
    f"{dataset_stem}_{timestamp}"
)

# 29. Configure monitoring output directory
REPORT_DIR = BASE_DIR / "reports" / "pipeline"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 30. Create consolidated monitoring report
summary = {
    "model_name": MODEL_NAME,
    "model_version": champion_version,
    "dataset": dataset_name,
    "samples": len(current),
    "baseline_mae": float(baseline_mae),
    "current_mae": float(current_mae),
    "current_rmse": float(current_rmse),
    "baseline_r2": float(baseline_r2),
    "current_r2": float(current_r2),
    "mae_increase_pct": float(
        mae_increase * 100
    ),
    "r2_decrease": float(r2_decrease),
    "data_drift": data_drift,
    "drifted_features": drifted_features,
    "prediction_ks": float(prediction_ks),
    "prediction_p_value": float(
        prediction_p_value
    ),
    "prediction_drift": prediction_drift,
    "residual_ks": float(residual_ks),
    "residual_p_value": float(
        residual_p_value
    ),
    "residual_drift": residual_drift,
    "mae_degraded": mae_degraded,
    "r2_degraded": r2_degraded,
    "concept_drift_suspected":
        concept_drift_suspected,
    "retraining_required": retraining_required,
    "decision": decision,
    "reason": reason,
    "feature_drift_results":
        feature_drift_results
}

# 31. Save consolidated JSON report
summary_path = (
    REPORT_DIR /
    f"{report_id}_summary.json"
)

with open(summary_path, "w") as file:

    json.dump(
        summary,
        file,
        indent=4
    )

# 32. Save feature drift report
feature_report_path = (
    REPORT_DIR /
    f"{report_id}_features.csv"
)

pd.DataFrame(
    feature_drift_results
).to_csv(
    feature_report_path,
    index=False
)

# 33. Save production prediction report
prediction_path = (
    REPORT_DIR /
    f"{report_id}_predictions.csv"
)

current.to_csv(
    prediction_path,
    index=False
)

# 34. Track monitoring pipeline in MLflow
with mlflow.start_run(
    run_name=f"monitoring_pipeline_{dataset_stem}"
) as run:

    # 34.1 Log monitoring configuration
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "model_version": champion_version,
        "dataset": dataset_name,
        "samples": len(current),
        "drift_threshold": DRIFT_THRESHOLD,
        "mae_threshold": MAE_THRESHOLD,
        "r2_threshold": R2_THRESHOLD
    })

    # 34.2 Log performance and drift metrics
    mlflow.log_metrics({
        "baseline_mae": float(baseline_mae),
        "mae": float(current_mae),
        "rmse": float(current_rmse),
        "r2": float(current_r2),
        "mae_increase_pct": float(
            mae_increase * 100
        ),
        "r2_decrease": float(r2_decrease),
        "drifted_feature_count": len(
            drifted_features
        ),
        "prediction_ks": float(prediction_ks),
        "prediction_p_value": float(
            prediction_p_value
        ),
        "residual_ks": float(residual_ks),
        "residual_p_value": float(
            residual_p_value
        )
    })

    # 34.3 Log monitoring decisions
    mlflow.set_tags({
        "training_stage": "automated_monitoring",
        "decision": decision,
        "data_drift": str(data_drift).lower(),
        "prediction_drift": str(
            prediction_drift
        ).lower(),
        "residual_drift": str(
            residual_drift
        ).lower(),
        "concept_drift_suspected": str(
            concept_drift_suspected
        ).lower(),
        "retraining_required": str(
            retraining_required
        ).lower()
    })

    # 34.4 Log monitoring reports
    for path in [
        summary_path,
        feature_report_path,
        prediction_path
    ]:

        mlflow.log_artifact(
            str(path),
            artifact_path="pipeline_reports"
        )

    print("\nMLflow Run ID:", run.info.run_id)

# 35. Display feature drift results
print("\nFeature Drift Results:")

print(
    pd.DataFrame(
        feature_drift_results
    ).to_string(index=False)
)

# 36. Display model performance
print("\nModel Performance:")

print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Current MAE: {current_mae:.2f}")
print(f"Current RMSE: {current_rmse:.2f}")
print(f"Current R2: {current_r2:.6f}")
print(f"MAE Increase: {mae_increase * 100:.2f}%")

# 37. Display monitoring indicators
print("\nMonitoring Indicators:")

print("Data Drift:", data_drift)
print("Drifted Features:", drifted_features)
print("Prediction Drift:", prediction_drift)
print("Residual Drift:", residual_drift)
print("MAE Degraded:", mae_degraded)
print("R2 Degraded:", r2_degraded)
print("Concept Drift:", concept_drift_suspected)

# 38. Display final monitoring decision
print("\nPipeline Decision:", decision)
print("Reason:", reason)
print("Retraining Required:", retraining_required)

# 39. Display retraining action
if retraining_required:

    print("\nRetraining request generated.")
    print("Collect and validate recent labelled data")
    print("before starting challenger training.")

else:

    print("\nNo automatic retraining initiated.")

# 40. Display report location
print("\nMonitoring report:", summary_path)

print("\nMonitoring pipeline completed!")