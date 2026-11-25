# 1. Import required libraries
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

# 2. Configure project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 3. Configure significance threshold
DRIFT_THRESHOLD = 0.05

# 4. Load baseline reference predictions
reference = pd.read_csv(
    BASE_DIR / "data" / "monitoring_reference.csv"
)

reference_predictions = reference["prediction"]

# 5. Define production prediction datasets
DATASETS = {
    "normal": "predictions_normal.csv",
    "data_drift": "predictions_data_drift.csv",
    "concept_drift": "predictions_concept_drift.csv"
}

# 6. Compare reference and current predictions
def detect_prediction_drift(reference, current):

    # 6.1 Perform KS Test
    statistic, p_value = ks_2samp(
        reference,
        current
    )

    # 6.2 Determine prediction drift
    drift_detected = p_value < DRIFT_THRESHOLD

    # 6.3 Return detection results
    return {
        "reference_mean": reference.mean(),
        "current_mean": current.mean(),
        "reference_std": reference.std(),
        "current_std": current.std(),
        "ks_statistic": statistic,
        "p_value": p_value,
        "prediction_drift": bool(drift_detected)
    }

# 7. Initialize storage for detection results
results = []

# 8. Evaluate all production scenarios
for scenario, filename in DATASETS.items():

    # 8.1 Load production predictions
    current = pd.read_csv(
        BASE_DIR / "data" / filename
    )

    # 8.2 Compare prediction distributions
    result = detect_prediction_drift(
        reference_predictions,
        current["prediction"]
    )

    # 8.3 Identify the scenario
    result["scenario"] = scenario

    # 8.4 Store detection results
    results.append(result)

# 9. Convert results into a DataFrame
results_df = pd.DataFrame(results)

# 10. Reorder report columns
results_df = results_df[
    [
        "scenario",
        "reference_mean",
        "current_mean",
        "reference_std",
        "current_std",
        "ks_statistic",
        "p_value",
        "prediction_drift"
    ]
]

# 11. Display prediction drift results
print("\nPrediction Drift Detection Results:")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

# 12. Save prediction drift report
results_df.to_csv(
    BASE_DIR / "reports" /
    "prediction_drift_report.csv",
    index=False
)

# 13. Display completion message
print("\nPrediction drift detection completed!")


