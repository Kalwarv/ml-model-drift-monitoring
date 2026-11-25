# 1. Import required libraries
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

# 2. Configure project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 3. Configure statistical significance threshold
DRIFT_THRESHOLD = 0.05

# 4. Load historical monitoring reference data
reference = pd.read_csv(
    BASE_DIR / "data" / "monitoring_reference.csv"
)

# 5. Define features to monitor
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

# 6. Define production scenarios
DATASETS = {
    "normal": "production_normal.csv",
    "data_drift": "production_data_drift.csv",
    "concept_drift": "production_concept_drift.csv"
}

# 7. Compare historical and production distributions
def detect_drift(reference, current, feature):

    # 7.1 Extract feature values
    reference_values = reference[feature]
    current_values = current[feature]

    # 7.2 Select the statistical test
    if feature == "bedrooms":

        frequency_table = pd.crosstab(
            pd.Series(
                ["reference"] * len(reference_values)
                + ["current"] * len(current_values)
            ),
            pd.concat(
                [reference_values, current_values],
                ignore_index=True
            )
        )

        statistic, p_value, _, _ = chi2_contingency(
            frequency_table
        )

        test_name = "Chi-square"

    else:

        statistic, p_value = ks_2samp(
            reference_values,
            current_values
        )

        test_name = "KS"
    # 7.3 Determine statistical drift
    drift_detected = p_value < DRIFT_THRESHOLD

    # 7.4 Return drift detection results
    return {
        "feature": feature,
        "test": test_name,
        "reference_mean": reference_values.mean(),
        "current_mean": current_values.mean(),
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": bool(drift_detected)
    }


# 8. Initialize storage for drift results
results = []

# 9. Evaluate each production scenario
for scenario, filename in DATASETS.items():

    # 9.1 Load current production data
    current = pd.read_csv(
        BASE_DIR / "data" / filename
    )

    # 9.2 Check each input feature
    for feature in FEATURES:

        result = detect_drift(
            reference,
            current,
            feature
        )

        # 9.3 Identify the production scenario
        result["scenario"] = scenario

        # 9.4 Store the drift result
        results.append(result)


# 10. Convert drift results into a DataFrame
results_df = pd.DataFrame(results)

# 11. Reorder report columns
results_df = results_df[
    [
        "scenario",
        "feature",
        "test",
        "reference_mean",
        "current_mean",
        "statistic",
        "p_value",
        "drift_detected"
    ]
]

# 12. Display statistical drift results
print("\nData Drift Detection Results:")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6g}"
    )
)

# 13. Save drift detection report
results_df.to_csv(
    BASE_DIR / "reports" / "data_drift_report.csv",
    index=False
)

# 14. Display drift summary
print("\nDrift Summary:")

print(
    results_df.groupby("scenario")[
        "drift_detected"
    ].sum()
)

print("\nData drift detection completed!")

