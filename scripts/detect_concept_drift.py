# 1. Import required libraries
import json
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

# 2. Configure project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 3. Configure monitoring thresholds
MAE_THRESHOLD = 0.25
R2_THRESHOLD = 0.10
DRIFT_THRESHOLD = 0.05

# 4. Load baseline performance metrics
with open(
    BASE_DIR / "reports" / "baseline_metrics.json"
) as file:
    baseline_metrics = json.load(file)

baseline_mae = baseline_metrics["mae"]
baseline_r2 = baseline_metrics["r2"]

# 5. Load historical monitoring reference data
reference = pd.read_csv(
    BASE_DIR / "data" / "monitoring_reference.csv"
)

# 5.1 Calculate reference residuals
reference_errors = (
    reference["price"] - reference["prediction"]
)

# 6. Load production evaluation metrics
production_metrics = pd.read_csv(
    BASE_DIR / "reports" /
    "production_evaluation.csv"
)

# 7. Define production prediction datasets
DATASETS = {
    "normal": "predictions_normal.csv",
    "data_drift": "predictions_data_drift.csv",
    "concept_drift": "predictions_concept_drift.csv"
}

# 8. Investigate potential concept drift
def detect_concept_drift(scenario, filename):

    # 8.1 Load production predictions
    current = pd.read_csv(
        BASE_DIR / "data" / filename
    )

    # 8.2 Extract current residuals
    current_errors = current["error"]

    # 8.3 Retrieve scenario performance metrics
    metrics = production_metrics[
        production_metrics["scenario"] == scenario
    ].iloc[0]

    current_mae = metrics["mae"]
    current_r2 = metrics["r2"]

    # 8.4 Calculate relative MAE increase
    mae_increase = (
        (current_mae - baseline_mae)
        / baseline_mae
    )

    # 8.5 Calculate absolute R2 decrease
    r2_decrease = baseline_r2 - current_r2

    # 8.6 Compare reference and current residuals
    statistic, p_value = ks_2samp(
        reference_errors,
        current_errors
    )

    # 8.7 Evaluate individual monitoring indicators
    mae_degraded = mae_increase > MAE_THRESHOLD

    r2_degraded = r2_decrease > R2_THRESHOLD

    residual_drift = p_value < DRIFT_THRESHOLD

    # 8.8 Raise potential concept drift alert
    concept_drift_suspected = (
        mae_degraded
        and (residual_drift or r2_degraded)
    )

    # 8.9 Return monitoring results
    return {
        "scenario": scenario,
        "baseline_mae": baseline_mae,
        "current_mae": current_mae,
        "mae_increase_pct": mae_increase * 100,
        "baseline_r2": baseline_r2,
        "current_r2": current_r2,
        "r2_decrease": r2_decrease,
        "residual_ks": statistic,
        "residual_p_value": p_value,
        "mae_degraded": bool(mae_degraded),
        "r2_degraded": bool(r2_degraded),
        "residual_drift": bool(residual_drift),
        "concept_drift_suspected": bool(
            concept_drift_suspected
        )
    }

# 9. Initialize storage for monitoring results
results = []

# 10. Evaluate all production scenarios
for scenario, filename in DATASETS.items():

    result = detect_concept_drift(
        scenario,
        filename
    )

    results.append(result)

# 11. Convert results into a DataFrame
results_df = pd.DataFrame(results)

# 12. Display monitoring results
print("\nConcept Drift Monitoring Results:")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

# 13. Save Concept Drift monitoring report
results_df.to_csv(
    BASE_DIR / "reports" /
    "concept_drift_report.csv",
    index=False
)

# 14. Display alert summary
print("\nConcept Drift Alert Summary:")

print(
    results_df[
        ["scenario", "concept_drift_suspected"]
    ].to_string(index=False)
)

print("\nConcept drift monitoring completed!")

