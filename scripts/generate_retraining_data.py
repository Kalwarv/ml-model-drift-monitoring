# 1. Import required libraries
import numpy as np
import pandas as pd
from pathlib import Path

# 2. Configure project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 3. Initialize reproducible random generator
rng = np.random.default_rng(200)

# 4. Configure dataset size
N_SAMPLES = 1000

# 5. Generate house features
house_size = rng.integers(
    800, 2201, size=N_SAMPLES
)

bedrooms = rng.integers(
    1, 5, size=N_SAMPLES
)

house_age = rng.integers(
    0, 31, size=N_SAMPLES
)

noise = rng.normal(
    0, 100000, size=N_SAMPLES
)

# 6. Generate prices under new market conditions
price = (
    house_size * 6000
    + bedrooms * 200000
    - house_age * 15000
    + noise
)

# 7. Create retraining DataFrame
df = pd.DataFrame({
    "house_size": house_size,
    "bedrooms": bedrooms,
    "house_age": house_age,
    "price": np.round(price, 2)
})

# 8. Save recent labelled data
output_path = (
    BASE_DIR / "data" /
    "retraining_data.csv"
)

df.to_csv(output_path, index=False)

# 9. Display dataset information
print("Retraining dataset generated!")

print("\nDataset shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe().round(2))

print("\nSaved file:", output_path)

