# 1. Import required libraries
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pyfunc

from pathlib import Path
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
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

# 3. Define input features
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

# 4. Load model acceptance report
acceptance = pd.read_csv(
    BASE_DIR / "reports" /
    "model_acceptance_report.csv"
).iloc[0]

# 5. Verify candidate acceptance
if str(acceptance["candidate_accepted"]).lower() != "true":

    raise RuntimeError(
        "Candidate rejected. Registration stopped."
    )

print("Candidate acceptance verified!")

# 6. Identify existing champion version
previous_champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

previous_version = previous_champion.version

print("Previous champion version:", previous_version)

# 7. Load accepted challenger model
challenger_model = joblib.load(
    BASE_DIR / "models" /
    "challenger_model.joblib"
)

print("Challenger model loaded!")

# 8. Load model comparison reports
comparison = pd.read_csv(
    BASE_DIR / "reports" /
    "retraining_comparison.csv"
)

historical_comparison = pd.read_csv(
    BASE_DIR / "reports" /
    "historical_model_comparison.csv"
)

champion_metrics = comparison[
    comparison["model"] == "champion"
].iloc[0]

challenger_metrics = comparison[
    comparison["model"] == "challenger"
].iloc[0]

historical_metrics = historical_comparison[
    historical_comparison["model"]
    == "historical_challenger"
].iloc[0]

# 9. Load recent data
df = pd.read_csv(
    BASE_DIR / "data" /
    "retraining_data.csv"
)

X = df[FEATURES]
y = df["price"]

# 10. Reproduce the original retraining split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 11. Revalidate challenger performance
challenger_predictions = challenger_model.predict(
    X_test
)

verified_mae = mean_absolute_error(
    y_test,
    challenger_predictions
)

verified_r2 = r2_score(
    y_test,
    challenger_predictions
)

# 12. Evaluate current champion on identical records
champion_model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{previous_version}"
)

current_champion_predictions = (
    champion_model.predict(X_test)
)

current_champion_mae = mean_absolute_error(
    y_test,
    current_champion_predictions
)

verified_improvement = (
    (current_champion_mae - verified_mae)
    / current_champion_mae
) * 100

# 13. Recheck acceptance before registration
if not (
    verified_improvement >= 20
    and verified_mae <= 150000
    and verified_r2 >= 0.95
):
    raise RuntimeError(
        "Candidate failed revalidation. "
        "Registration stopped."
    )

print("Candidate performance revalidated!")

# 14. Prepare model signature and input example
signature = infer_signature(
    X_test,
    challenger_predictions
)

input_example = X_test.head(3)

# 15. Start MLflow retraining tracking run
with mlflow.start_run(
    run_name="challenger_retraining"
) as run:

    # 15.1 Log retraining parameters
    mlflow.log_params({
        "algorithm": "LinearRegression",
        "previous_champion_version": previous_version,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "feature_count": len(FEATURES),
        "test_size": 0.2,
        "random_state": 42,
        "retraining_reason": "concept_drift",
        "min_mae_improvement": 20,
        "max_challenger_mae": 150000,
        "min_challenger_r2": 0.95
    })

    # 15.2 Log challenger performance metrics
    mlflow.log_metrics({
        "mae": float(verified_mae),
        "rmse": float(challenger_metrics["rmse"]),
        "r2": float(verified_r2),
        "champion_mae": float(current_champion_mae),
        "mae_improvement_pct": float(
            verified_improvement
        ),
        "historical_mae": float(
            historical_metrics["mae"]
        ),
        "historical_r2": float(
            historical_metrics["r2"]
        )
    })

    # 15.3 Log learned model coefficients
    for feature, coefficient in zip(
        FEATURES,
        challenger_model.coef_
    ):

        mlflow.log_metric(
            f"coefficient_{feature}",
            float(coefficient)
        )

    mlflow.log_metric(
        "intercept",
        float(challenger_model.intercept_)
    )

    # 15.4 Add descriptive run tags
    mlflow.set_tags({
        "training_stage": "retraining",
        "drift_scenario": "concept_drift",
        "task_type": "regression",
        "dataset_type": "synthetic",
        "candidate_accepted": "true",
        "promotion_status": "pending"
    })

    # 15.5 Log evaluation report artifacts
    REPORTS = [
        "retraining_comparison.csv",
        "historical_model_comparison.csv",
        "model_acceptance_report.csv",
        "concept_drift_report.csv"
    ]

    for report_name in REPORTS:

        mlflow.log_artifact(
            str(
                BASE_DIR / "reports" /
                report_name
            ),
            artifact_path="evaluation_reports"
        )

    # 15.6 Log challenger model to MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=challenger_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # 15.7 Register the accepted challenger
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=MODEL_NAME
    )

    new_version = registered_model.version

    print("\nNew model version:", new_version)

    # 15.8 Verify registered model inference
    registered_candidate = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/{new_version}"
    )

    registered_predictions = (
        registered_candidate.predict(X_test)
    )

    # 15.9 Confirm predictions survived serialization
    np.testing.assert_allclose(
        challenger_predictions,
        registered_predictions,
        rtol=1e-8,
        atol=1e-6
    )

    print("Registered model inference verified!")

    # 15.10 Assign challenger alias
    client.set_registered_model_alias(
        MODEL_NAME,
        "challenger",
        new_version
    )

    # 15.11 Ensure champion has not changed
    active_champion = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            "champion"
        )
    )

    if active_champion.version != previous_version:

        raise RuntimeError(
            "Champion changed during evaluation. "
            "Promotion stopped."
        )

    # 15.12 Preserve old champion for rollback
    client.set_registered_model_alias(
        MODEL_NAME,
        "previous_champion",
        previous_version
    )

    # 15.13 Promote accepted candidate
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        new_version
    )

    # 15.14 Remove challenger alias after promotion
    client.delete_registered_model_alias(
        MODEL_NAME,
        "challenger"
    )

    # 15.15 Record successful promotion
    mlflow.set_tag(
        "promotion_status",
        "promoted"
    )

    mlflow.set_tag(
        "registered_version",
        str(new_version)
    )

    print("\nMLflow Run ID:", run.info.run_id)

# 16. Verify final champion alias
final_champion = (
    client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )
)

assert final_champion.version == new_version

# 17. Display promotion summary
print("\nModel Promotion Summary:")
print("Registered model:", MODEL_NAME)
print("Previous champion:", previous_version)
print("New champion:", final_champion.version)
print("Champion MAE:", round(verified_mae, 2))
print("Champion R2:", round(verified_r2, 6))

print(
    "MAE improvement:",
    round(verified_improvement, 2),
    "%"
)

# 18. Save promotion report
promotion_df = pd.DataFrame([{
    "model_name": MODEL_NAME,
    "previous_version": previous_version,
    "new_version": new_version,
    "mae": verified_mae,
    "r2": verified_r2,
    "mae_improvement_pct": verified_improvement,
    "promotion_status": "promoted"
}])

promotion_df.to_csv(
    BASE_DIR / "reports" /
    "model_promotion_report.csv",
    index=False
)

print("\nModel registration and promotion completed!")