import pandas as pd
import numpy as np
import os

# Make results reproducible
np.random.seed(42)

# Number of machine records
num_samples = 1000

# Generate sensor values
temperature = np.random.randint(40, 95, num_samples)          # °C
vibration = np.round(np.random.uniform(0.5, 8.0, num_samples), 2)  # mm/s
current = np.round(np.random.uniform(3.0, 9.5, num_samples), 2)    # Ampere
rpm = np.random.randint(1100, 1501, num_samples)              # RPM
operating_hours = np.random.randint(50, 5001, num_samples)    # Hours

# Determine machine condition
status = []

for t, v, c, r, h in zip(temperature, vibration, current, rpm, operating_hours):

    if (
        t >= 80
        or v >= 5.5
        or c >= 8.0
        or r <= 1250
        or h >= 4000
    ):
        status.append("Failure")

    elif (
        t >= 65
        or v >= 3.5
        or c >= 6.5
        or r <= 1350
        or h >= 2500
    ):
        status.append("Warning")

    else:
        status.append("Normal")

# Create DataFrame
df = pd.DataFrame({
    "Temperature": temperature,
    "Vibration": vibration,
    "Current": current,
    "RPM": rpm,
    "Operating_Hours": operating_hours,
    "Status": status
})

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save dataset
output_path = "data/machine_data.csv"
df.to_csv(output_path, index=False)

print("=" * 50)
print("AI Predictive Maintenance Dataset Generated")
print("=" * 50)
print(f"Total Records : {len(df)}")
print(f"Saved To      : {output_path}")
print("\nFirst 5 Records:\n")
print(df.head())

print("\nMachine Status Distribution:\n")
print(df["Status"].value_counts())