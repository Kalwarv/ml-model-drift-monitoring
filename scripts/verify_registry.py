import mlflow
import mlflow.pyfunc #PyFunc (Python Function) provides a common interface for loading and using ML models, regardless of whether they were trained using Scikit-learn, PyTorch or another supported framework.
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient

#Configure MLflow
BASE_DIR = Path(__file__).resolve().parent.parent
TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)

#Retrieve the champion model
client = MlflowClient()
MODEL_NAME = "HousePriceDriftModel"

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    "champion"
)

#Print model information
print("Model:", MODEL_NAME)
print("Version:", champion.version)
print("Status:", champion.status)
print("Run ID:", champion.run_id)

#Load the model
model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

#Prepare sample input
sample = pd.DataFrame({
    "house_size": [1500],
    "bedrooms": [3],
    "house_age": [10]
})

#Generate a prediction
prediction = model.predict(sample)

print("\nInput:\n", sample)
print(f"Predicted price: Rs. {prediction[0]:,.2f}")

