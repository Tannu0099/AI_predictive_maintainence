import joblib
import pandas as pd

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/predictive_model.pkl")

# Load Label Encoder
encoder = joblib.load("models/label_encoder.pkl")

print("=" * 50)
print("AI Predictive Maintenance System")
print("=" * 50)

# -----------------------------
# User Input
# -----------------------------
temperature = float(input("Enter Temperature (°C): "))
vibration = float(input("Enter Vibration (mm/s): "))
current = float(input("Enter Motor Current (A): "))
rpm = int(input("Enter RPM: "))
operating_hours = int(input("Enter Operating Hours: "))

# -----------------------------
# Create DataFrame
# -----------------------------
new_data = pd.DataFrame({
    "Temperature": [temperature],
    "Vibration": [vibration],
    "Current": [current],
    "RPM": [rpm],
    "Operating_Hours": [operating_hours]
})

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(new_data)

# Convert numeric label back to text
status = encoder.inverse_transform(prediction)[0]

# Prediction Probability
probability = model.predict_proba(new_data)

confidence = max(probability[0]) * 100

print("\n" + "=" * 50)
print("Prediction Result")
print("=" * 50)

print(f"Machine Status : {status}")
print(f"Confidence     : {confidence:.2f}%")

# -----------------------------
# Maintenance Recommendation
# -----------------------------
print("\nMaintenance Recommendation:")

if status == "Normal":
    print("✅ Machine is operating normally.")
    print("✔ No maintenance required.")

elif status == "Warning":
    print("⚠ Machine requires inspection.")
    print("✔ Check vibration and temperature.")
    print("✔ Schedule preventive maintenance.")

elif status == "Failure":
    print("❌ High risk of machine failure.")
    print("✔ Stop machine if necessary.")
    print("✔ Inspect motor, bearings, and electrical system.")
    print("✔ Perform immediate maintenance.")

print("=" * 50)