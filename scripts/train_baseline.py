import json
import numpy as np
import pandas as pd
import joblib
import mlflow 
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

EXPERIMENT_NAME = "HousePriceDriftMonitoring"
MODEL_NAME = "HousePriceDriftModel"
mlflow.set_experiment(EXPERIMENT_NAME)

# 2. Load reference dataset
df = pd.read_csv(
    BASE_DIR / "data" / "reference_data.csv"
)

# 3. Separate features and target
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

X = df[FEATURES]
y = df["price"]

# 4. Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

run = mlflow.start_run(run_name="baseline_linear_regression")
print("MLflow Run ID:", run.info.run_id)

mlflow.log_param("algorithm", "LinearRegression")
mlflow.log_param("test_size", 0.2)
mlflow.log_param("random_state", 42)
mlflow.log_param("training_samples", len(X_train))
mlflow.log_param("testing_samples", len(X_test))
mlflow.log_param("feature_count", len(FEATURES))

mlflow.set_tag("training_stage", "baseline")
mlflow.set_tag("drift_scenario", "none")
mlflow.set_tag("task_type", "regression")
mlflow.set_tag("dataset_type", "synthetic")

# 5. Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Generate predictions
predictions = model.predict(X_test)

# 7. Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
r2 = r2_score(y_test, predictions)
mlflow.log_metric("mae", mae)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)

metrics = {
    "model": "LinearRegression",
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "mae": float(mae),
    "rmse": float(rmse),
    "r2": float(r2)
}

# 8. Save trained model
model_path = BASE_DIR / "models" / "baseline_model.joblib"

joblib.dump(model, model_path)

# 9. Save monitoring reference dataset
reference = X_test.copy()
reference["price"] = y_test
reference["prediction"] = predictions

reference.to_csv(
    BASE_DIR / "data" / "monitoring_reference.csv",
    index=False
)

# 10. Save baseline metrics
metrics_path = BASE_DIR / "reports" / "baseline_metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)

mlflow.log_artifact(
    str(metrics_path),
    artifact_path="reports"
)

mlflow.log_artifact(
    str(BASE_DIR / "data" / "monitoring_reference.csv"),
    artifact_path="data"
)

# Model signature and input example
signature = infer_signature(X_test, predictions)
input_example = X_test.head(3)

# Log trained model to MLflow
model_info = mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    signature=signature,
    input_example=input_example
)

# Register model
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=MODEL_NAME
)

model_version = registered_model.version

# Configure Model Registry client
client = MlflowClient()

# Check existing model aliases
aliases = client.get_registered_model(MODEL_NAME).aliases

# Assign champion only if one does not exist
if "champion" not in aliases:
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        model_version
    )

# 11. Display results
print("Baseline model trained successfully!")

print("\nDataset Split:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nLearned Model Coefficients:")
for feature, coefficient in zip(FEATURES, model.coef_):
    print(f"{feature}: {coefficient:,.2f}")

print(f"Intercept: {model.intercept_:,.2f}")

print("\nBaseline Model Performance:")
print(f"MAE:  Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2:   {r2:.6f}")

print("\nSample Predictions:")
print(
    reference[
        ["house_size", "price", "prediction"]
    ].head().round(2)
)

print("\nSaved files:")
print(model_path)
print(BASE_DIR / "data" / "monitoring_reference.csv")
print(metrics_path)

# Display MLflow registration details
print("\nMLflow Information:")
print("Experiment:", EXPERIMENT_NAME)
print("Run ID:", run.info.run_id)
print("Registered Model:", MODEL_NAME)
print("Registered Version:", model_version)

champion = client.get_model_version_by_alias(
    MODEL_NAME, "champion"
)

print("Champion Version:", champion.version)

# End MLflow run
mlflow.end_run()