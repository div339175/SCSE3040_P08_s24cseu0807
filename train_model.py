import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Generate 600 realistic delivery records (as specified in Practical 07 & 08)
n = 600
distance_km = np.random.uniform(0.5, 30.0, n)
prep_time_min = np.random.uniform(5.0, 45.0, n)
traffic_level = np.random.choice([1, 2, 3], size=n, p=[0.3, 0.5, 0.2])
rain = np.random.choice([0, 1], size=n, p=[0.8, 0.2])

# Ground truth formula with small Gaussian noise
# delivery_time_min approx 10 + 2.0*distance + 1.0*prep + 4.0*traffic + 6.0*rain
y = 10.0 + 2.0 * distance_km + 1.0 * prep_time_min + 3.5 * traffic_level + 5.0 * rain + np.random.normal(0, 1.5, n)

X = pd.DataFrame({
    'distance_km': distance_km,
    'prep_time_min': prep_time_min,
    'traffic_level': traffic_level,
    'rain': rain
})

model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.joblib')
print("Model trained and saved to model.joblib successfully.")

# Test benchmark sample
test_order = pd.DataFrame([{
    'distance_km': 5.0,
    'prep_time_min': 20.0,
    'traffic_level': 2,
    'rain': 0
}])
pred = model.predict(test_order)[0]
print(f"Sample prediction for [5km, 20min prep, traffic 2, rain 0]: {pred:.2f} minutes")
