# 1. Import required libraries
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

from pathlib import Path
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
# Configure experiment for production monitoring
EXPERIMENT_NAME = "HousePriceDriftMonitoring"
mlflow.set_experiment(EXPERIMENT_NAME)

# 3. Specify the registered model and fixed version
MODEL_NAME = "HousePriceDriftModel"
MODEL_VERSION = "1"

# 4. Load the registered model from MLflow Registry
# Use a fixed version to ensure consistent drift evaluation
model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

print("Loaded model version:", MODEL_VERSION)

# 5. Define the input features expected by the model
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

# 6. Define production datasets for each scenario
DATASETS = {
    "normal": "production_normal.csv",
    "data_drift": "production_data_drift.csv",
    "concept_drift": "production_concept_drift.csv"
}

# 7. Create a reusable production evaluation function
def evaluate_dataset(scenario, filename):

    # 7.1 Load the production dataset
    df = pd.read_csv(
        BASE_DIR / "data" / filename
    )

    # 7.2 Separate input features and actual target values
    X = df[FEATURES]
    y = df["price"]

    # 7.3 Generate predictions using the frozen model
    predictions = model.predict(X)

    # 7.4 Calculate model performance metrics
    mae = mean_absolute_error(y, predictions)

    rmse = np.sqrt(
        mean_squared_error(y, predictions)
    )

    r2 = r2_score(y, predictions)

    # 7.5 Store predictions in the dataset
    df["prediction"] = predictions

    # 7.6 Calculate residuals (actual - predicted)
    df["error"] = df["price"] - df["prediction"]

    # 7.7 Calculate absolute prediction errors
    df["absolute_error"] = df["error"].abs()

    # 7.8 Define the output path for prediction logging
    output_path = (
        BASE_DIR / "data" /
        f"predictions_{scenario}.csv"
    )

    # 7.9 Save inputs, actual values, predictions and errors
    df.to_csv(output_path, index=False)

    # 7.10 Return evaluation metrics for comparison
    return {
        "scenario": scenario,
        "samples": len(df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_actual": y.mean(),
        "mean_prediction": predictions.mean(),
        "mean_error": df["error"].mean()
    }


# 8. Initialize a list to store evaluation results
results = []

# 9. Evaluate all production datasets
for scenario, filename in DATASETS.items():

    # 9.1 Start an independent MLflow monitoring run
    with mlflow.start_run(
        run_name=f"evaluation_{scenario}"
    ) as run:

        # 9.2 Evaluate the current production scenario
        result = evaluate_dataset(
            scenario,
            filename
        )

        # 9.3 Store the evaluation results
        results.append(result)

        # 9.4 Log monitoring parameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "dataset": filename,
            "samples": result["samples"],
            "feature_count": len(FEATURES)
        })       
        # 9.5 Log production performance metrics
        mlflow.log_metrics({
            "mae": float(result["mae"]),
            "rmse": float(result["rmse"]),
            "r2": float(result["r2"]),
            "mean_actual": float(result["mean_actual"]),
            "mean_prediction": float(result["mean_prediction"]),
            "mean_error": float(result["mean_error"])
        })
        # 9.6 Add descriptive monitoring tags
        mlflow.set_tags({
            "training_stage": "evaluation",
            "drift_scenario": scenario,
            "task_type": "regression",
            "dataset_type": "synthetic"
        })
        # 9.7 Identify the production prediction file
        prediction_path = (
            BASE_DIR / "data" /
            f"predictions_{scenario}.csv"
        )

        # 9.8 Log production predictions as an artifact
        mlflow.log_artifact(
            str(prediction_path),
            artifact_path="predictions"
        )
        # 9.9 Display monitoring run information
        print(f"\nScenario: {scenario}")
        print("MLflow Run ID:", run.info.run_id)
        print("Evaluation metrics logged successfully")
        

# 10. Convert evaluation results into a DataFrame
results_df = pd.DataFrame(results)

# 11. Display the production evaluation results
print("\nProduction Evaluation Results:")

print(
    results_df.round({
        "mae": 2,
        "rmse": 2,
        "r2": 6,
        "mean_actual": 2,
        "mean_prediction": 2,
        "mean_error": 2
    }).to_string(index=False)
)

# 12. Save the combined evaluation report
results_df.to_csv(
    BASE_DIR / "reports" /
    "production_evaluation.csv",
    index=False
)

# 13. Display completion message
print("\nProduction evaluation completed!")