# predict.py
import joblib
import numpy as np

# Load the compressed model file
model_bundle = joblib.load("liver_model_compressed.pkl")

model = model_bundle["model"]
scaler = model_bundle["scaler"]
imputer = model_bundle["imputer"]
threshold = model_bundle["threshold"]
feature_columns = model_bundle["feature_columns"]

print("🔹 Enter patient details:")

# Collect user input
patient_data = {}
for col in feature_columns:
    value = input(f"{col}: ")
    try:
        patient_data[col] = float(value)
    except ValueError:
        patient_data[col] = np.nan  # handle missing input

# Convert to array in correct order
X = np.array([[patient_data[col] for col in feature_columns]])

# Preprocess: impute missing + scale
X_imputed = imputer.transform(X)
X_scaled = scaler.transform(X_imputed)

# Predict probability
prob = model.predict_proba(X_scaled)[0, 1]

# Apply threshold
pred = int(prob >= threshold)

# Confidence %
confidence = round(prob * 100, 2)

# Map prediction
label = "Positive (Liver Disease)" if pred == 1 else "Negative (Healthy)"

print(f"\n✅ Prediction: {label}")
print(f"🔎 Confidence: {confidence}% (Threshold={threshold:.2f})")

