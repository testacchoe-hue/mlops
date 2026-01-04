import joblib
import numpy as np

# Load the trained model artifact
model_path = "artifacts/classification_model.pkl"
model = joblib.load(model_path)

print("✅ Model loaded successfully")

# Create a dummy input sample
# Number of features must match the training data (breast cancer dataset = 30 features)
sample_input = np.random.rand(1, 30)

# Run prediction
prediction = model.predict(sample_input)
prediction_proba = model.predict_proba(sample_input)

print("✅ Prediction output:", prediction)
print("✅ Prediction probability:", prediction_proba)

print("🎉 Model is working correctly inside the virtual environment")

