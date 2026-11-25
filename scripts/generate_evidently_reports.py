# 1. Import required libraries
import pandas as pd
from pathlib import Path

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    RegressionPreset
)


# 2. Configure project directories
BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports" / "evidently"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# 3. Define columns required for monitoring
FEATURES = [
    "house_size",
    "bedrooms",
    "house_age"
]

COLUMNS = FEATURES + [
    "price",
    "prediction"
]

# 4. Configure Evidently column mapping
column_mapping = ColumnMapping()

column_mapping.target = "price"
column_mapping.prediction = "prediction"

column_mapping.numerical_features = [
    "house_size",
    "house_age"
]

column_mapping.categorical_features = [
    "bedrooms"
]

column_mapping.task = "regression"

# 5. Load historical monitoring reference data
reference = pd.read_csv(
    BASE_DIR / "data" /
    "monitoring_reference.csv"
)

reference = reference[COLUMNS]

# 6. Define production prediction datasets
DATASETS = {
    "normal": "predictions_normal.csv",
    "data_drift": "predictions_data_drift.csv",
    "concept_drift": "predictions_concept_drift.csv"
}

# 7. Generate Evidently monitoring reports
def generate_report(scenario, filename):

    # 7.1 Load current production data
    current = pd.read_csv(
        BASE_DIR / "data" / filename
    )

    current = current[COLUMNS]

    # 7.2 Configure monitoring presets
    report = Report(
        metrics=[
            DataDriftPreset(),
            RegressionPreset()
        ]
    )

    # 7.3 Execute monitoring analysis
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping
    )

    # 7.4 Define report output paths
    html_path = (
        REPORT_DIR /
        f"{scenario}_report.html"
    )

    json_path = (
        REPORT_DIR /
        f"{scenario}_report.json"
    )

    # 7.5 Save the monitoring reports
    report.save_html(str(html_path))
    report.save_json(str(json_path))

    # 7.6 Display generated report paths
    print(f"\nScenario: {scenario}")
    print("HTML report:", html_path)
    print("JSON report:", json_path)

# 8. Generate reports for all scenarios
for scenario, filename in DATASETS.items():

    generate_report(
        scenario,
        filename
    )

# 9. Display completion message
print("\nAll Evidently reports generated successfully!")