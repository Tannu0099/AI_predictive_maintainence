import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Reproducible data
np.random.seed(42)

# Number of training samples
n = 2000


# Generate synthetic sensor data
temperature = np.random.uniform(40, 100, n)
vibration = np.random.uniform(0.5, 8, n)
current = np.random.uniform(3, 20, n)
rpm = np.random.uniform(800, 3000, n)


# Create labels
def get_status(temp, vib, curr):

    if temp >= 85 or vib >= 6 or curr >= 16:
        return "Critical"

    elif temp >= 70 or vib >= 4 or curr >= 12:
        return "Warning"

    else:
        return "Healthy"


status = [
    get_status(t, v, c)
    for t, v, c in zip(temperature, vibration, current)
]


# Create dataframe
df = pd.DataFrame({
    "temperature": temperature,
    "vibration": vibration,
    "current": current,
    "rpm": rpm,
    "status": status
})


# Features
X = df[
    [
        "temperature",
        "vibration",
        "current",
        "rpm"
    ]
]

# Target
y = df["status"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Test
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Model Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)


# Save model
joblib.dump(
    model,
    "models/predictive_maintenance_model.pkl"
)


print("\nModel saved successfully.")