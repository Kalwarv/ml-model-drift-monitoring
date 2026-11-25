import numpy as np
import pandas as pd
from pathlib import Path

# 1. Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Random number generator
rng = np.random.default_rng(42)

# 3. Number of houses
n_samples = 1000

# 4. Generate input features
house_size = rng.integers(
    800, 2201, size=n_samples
)

bedrooms = rng.integers(
    1, 5, size=n_samples
)

house_age = rng.integers(
    0, 31, size=n_samples
)

# 5. Generate random price variation
noise = rng.normal(
    loc=0,
    scale=100000,
    size=n_samples
)

# 6. Calculate house prices
price = (
    house_size * 4000
    + bedrooms * 200000
    - house_age * 15000
    + noise
)

# 7. Create DataFrame
df = pd.DataFrame({
    "house_size": house_size,
    "bedrooms": bedrooms,
    "house_age": house_age,
    "price": np.round(price, 2)
})

# 8. Save dataset
output_path = BASE_DIR / "data" / "reference_data.csv"

df.to_csv(output_path, index=False)

# 9. Verify generated data
print("Reference dataset generated successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe().round(2))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset saved at:")
print(output_path)
