import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(100)

def generate_data(
    n_samples,
    min_size,
    max_size,
    price_per_sqft
):
    house_size = rng.integers(
        min_size, max_size + 1, size=n_samples
    )

    bedrooms = rng.integers(1, 5, size=n_samples)
    house_age = rng.integers(0, 31, size=n_samples)
    noise = rng.normal(0, 100000, size=n_samples)

    price = (
        house_size * price_per_sqft
        + bedrooms * 200000
        - house_age * 15000
        + noise
    )

    return pd.DataFrame({
        "house_size": house_size,
        "bedrooms": bedrooms,
        "house_age": house_age,
        "price": np.round(price, 2)
    })

# Scenario A: Normal production
normal_data = generate_data(
    n_samples=200,
    min_size=800,
    max_size=2200,
    price_per_sqft=4000
)
#Save the dataset
normal_data.to_csv(
    BASE_DIR / "data" / "production_normal.csv",
    index=False
)

# Scenario B: Data Drift
data_drift = generate_data(
    n_samples=200,
    min_size=2200,
    max_size=3600,
    price_per_sqft=4000
)

data_drift.to_csv(
    BASE_DIR / "data" / "production_data_drift.csv",
    index=False
)

# Scenario C: Concept Drift
concept_drift = generate_data(
    n_samples=200,
    min_size=800,
    max_size=2200,
    price_per_sqft=6000
)

concept_drift.to_csv(
    BASE_DIR / "data" / "production_concept_drift.csv",
    index=False
)

datasets = {
    "Normal Production": normal_data,
    "Data Drift": data_drift,
    "Concept Drift": concept_drift
}

for name, dataset in datasets.items():
    print(f"\n{'=' * 40}")
    print(name)
    print(f"{'=' * 40}")

    print("Dataset shape:", dataset.shape)

    print("\nMean values:")
    print(dataset.mean().round(2))

    print("\nHouse size range:")
    print("Minimum:", dataset["house_size"].min())
    print("Maximum:", dataset["house_size"].max())

print("\nAll production datasets generated!")

