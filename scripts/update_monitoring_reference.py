# 1. Import required libraries
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

from pathlib import Path
from mlflow import MlflowClient
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

client = MlflowClient()

# 3. Identify current champion version
champion_version = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
).version

print("Current champion version:", champion_version)

# 4. Verify the expected champion
if str(champion_version) != "2":
    raise RuntimeError(
        "This reference generation script expects Version 2."
    )

# 5. Load current champion model
model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{champion_version}"
)

# 6. Define model features
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

# 7. Load recent labelled dataset
df = pd.read_csv(
    BASE_DIR / "data" / "retraining_data.csv"
)

X = df[FEATURES]
y = df["price"]

# 8. Reproduce the retraining holdout split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 9. Generate champion predictions
predictions = model.predict(X_test)

# 10. Create version-specific monitoring reference
reference = X_test.copy()

reference["price"] = y_test.to_numpy()
reference["prediction"] = predictions

reference["error"] = (
    reference["price"] - reference["prediction"]
)

reference["absolute_error"] = (
    reference["error"].abs()
)

# 11. Define reference output path
reference_path = (
    BASE_DIR / "data" /
    f"monitoring_reference_v{champion_version}.csv"
)

# 12. Save new monitoring reference
reference.to_csv(
    reference_path,
    index=False
)

# 13. Calculate new baseline performance metrics
metrics = {
    "model_name": MODEL_NAME,
    "model_version": champion_version,
    "samples": len(reference),
    "mae": float(
        mean_absolute_error(y_test, predictions)
    ),
    "rmse": float(
        np.sqrt(
            mean_squared_error(y_test, predictions)
        )
    ),
    "r2": float(
        r2_score(y_test, predictions)
    )
}

# 14. Define version-specific baseline metrics path
metrics_path = (
    BASE_DIR / "reports" /
    f"baseline_metrics_v{champion_version}.json"
)

# 15. Save new baseline metrics
with open(metrics_path, "w") as file:
    json.dump(
        metrics,
        file,
        indent=4
    )

# 16. Display reference information
print("\nNew Monitoring Reference:")

print("Model version:", champion_version)
print("Reference samples:", len(reference))

print("\nNew Baseline Metrics:")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nReference saved:", reference_path)
print("Metrics saved:", metrics_path)

# 17. Display completion message
print("\nMonitoring reference updated successfully!")