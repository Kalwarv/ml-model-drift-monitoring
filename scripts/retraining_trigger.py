# 1. Import required libraries
import mlflow.pyfunc
import mlflow.sklearn
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import mlflow

from pathlib import Path
from datetime import datetime, timezone
from mlflow import MlflowClient
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

MODEL_NAME = "HousePriceDriftModel"

mlflow.set_experiment("HousePriceDriftMonitoring")

client = MlflowClient()

# 3. Configure command-line arguments
parser = argparse.ArgumentParser(
    description="Conditional model retraining workflow"
)

parser.add_argument(
    "--report",
    required=True,
    help="Monitoring summary JSON file"
)

parser.add_argument(
    "--training-data",
    default=None,
    help="Approved recent labelled CSV file"
)

args = parser.parse_args()

# 4. Load monitoring report
report_path = Path(args.report)

with open(report_path) as file:
    monitoring_report = json.load(file)

# 5. Identify active champion version
champion_version = str(
    client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    ).version
)

print("Current champion:", champion_version)

# 6. Verify monitoring report belongs to active model
if monitoring_report["model_name"] != MODEL_NAME:
    raise RuntimeError("Monitoring model mismatch.")

if str(monitoring_report["model_version"]) != champion_version:
    raise RuntimeError(
        "Monitoring report is stale. "
        "Run monitoring against the current champion."
    )

# 7. Check whether retraining is required
if monitoring_report["retraining_required"] is not True:

    print("\nRetraining Required: False")
    print("Action: Continue monitoring.")
    raise SystemExit(0)

print("\nRetraining Required: True")
print("Reason:", monitoring_report["reason"])

# 8. Configure retraining request directory
REQUEST_DIR = BASE_DIR / "reports" / "retraining_requests"

REQUEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 9. Generate request identifier from monitoring report
request_id = hashlib.sha256(
    report_path.read_bytes()
).hexdigest()[:16]

request_path = (
    REQUEST_DIR /
    f"request_{request_id}.json"
)

# 10. Create retraining request
request = {
    "request_id": request_id,
    "created_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "model_name": MODEL_NAME,
    "champion_version": champion_version,
    "monitoring_report": str(report_path),
    "reason": monitoring_report["reason"],
    "status": "WAITING_FOR_TRAINING_DATA"
}

# 11. Create helper function to save request
def save_request():

    with open(request_path, "w") as file:

        json.dump(
            request,
            file,
            indent=4
        )


# 12. Check availability of approved training data
if args.training_data is None:

    save_request()

    print("\nRetraining request saved:", request_path)
    print("Status: WAITING_FOR_TRAINING_DATA")
    print("No model has been trained or promoted.")

    raise SystemExit(0)

# 13. Load approved recent labelled data
training_path = (
    BASE_DIR / "data" / args.training_data
)

df = pd.read_csv(training_path)

# 14. Define expected features
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

REQUIRED_COLUMNS = FEATURES + ["price"]

# 15. Validate required columns
missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# 16. Validate dataset size and numerical values
if len(df) < 500:
    raise ValueError(
        "At least 500 training records required."
    )

if df[REQUIRED_COLUMNS].isnull().any().any():
    raise ValueError("Missing values detected.")

if not np.isfinite(
    df[REQUIRED_COLUMNS].to_numpy(dtype=float)
).all():
    raise ValueError("Invalid numerical values detected.")

# 17. Separate features and target
X = df[FEATURES]
y = df["price"]

# 18. Create training and holdout datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 19. Load the existing champion
champion = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{champion_version}"
)

# 20. Train challenger
challenger = LinearRegression()

challenger.fit(
    X_train,
    y_train
)

# 21. Generate predictions on identical holdout data
champion_predictions = champion.predict(X_test)

challenger_predictions = challenger.predict(X_test)

# 22. Calculate champion performance
champion_mae = mean_absolute_error(
    y_test,
    champion_predictions
)

# 23. Calculate challenger performance
challenger_mae = mean_absolute_error(
    y_test,
    challenger_predictions
)

challenger_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        challenger_predictions
    )
)

challenger_r2 = r2_score(
    y_test,
    challenger_predictions
)

# 24. Calculate relative model improvement
mae_improvement = (
    (
        champion_mae - challenger_mae
    ) / champion_mae
) * 100 if champion_mae > 0 else 0

# 25. Apply model acceptance gate
candidate_accepted = bool(
    mae_improvement >= 20
    and challenger_mae <= 150000
    and challenger_r2 >= 0.95
)

# 26. Track conditional retraining experiment
with mlflow.start_run(
    run_name="conditional_retraining"
) as run:

    mlflow.log_params({
        "request_id": request_id,
        "champion_version": champion_version,
        "training_dataset": args.training_data,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "algorithm": "LinearRegression"
    })

    mlflow.log_metrics({
        "champion_mae": float(champion_mae),
        "challenger_mae": float(challenger_mae),
        "challenger_rmse": float(challenger_rmse),
        "challenger_r2": float(challenger_r2),
        "mae_improvement_pct": float(
            mae_improvement
        )
    })

    mlflow.set_tags({
        "training_stage": "conditional_retraining",
        "candidate_accepted": str(
            candidate_accepted
        ).lower(),
        "promotion_status": "not_promoted"
    })

    # 27. Log accepted candidate without promotion
    if candidate_accepted:

        signature = mlflow.models.infer_signature(
            X_test,
            challenger_predictions
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=challenger,
            artifact_path="candidate_model",
            signature=signature,
            input_example=X_test.head(3)
        )

        request["candidate_model_uri"] = (
            model_info.model_uri
        )

    # 28. Update retraining request
    request.update({
        "training_dataset": args.training_data,
        "mlflow_run_id": run.info.run_id,
        "champion_mae": float(champion_mae),
        "challenger_mae": float(challenger_mae),
        "challenger_r2": float(challenger_r2),
        "mae_improvement_pct": float(
            mae_improvement
        ),
        "candidate_accepted": candidate_accepted,
        "status": (
            "READY_FOR_APPROVAL"
            if candidate_accepted
            else "CANDIDATE_REJECTED"
        )
    })

    save_request()

    mlflow.log_artifact(
        str(request_path),
        artifact_path="retraining_requests"
    )

# 29. Display evaluation summary
print("\nConditional Retraining Results:")

print(f"Champion MAE: {champion_mae:.2f}")
print(f"Challenger MAE: {challenger_mae:.2f}")
print(f"Challenger R2: {challenger_r2:.6f}")
print(f"MAE Improvement: {mae_improvement:.2f}%")

print("Candidate Accepted:", candidate_accepted)

print("Request Status:", request["status"])

print("\nChampion remains unchanged:", champion_version)

print("Request saved:", request_path)

print("\nConditional retraining workflow completed!")