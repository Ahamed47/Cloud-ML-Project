import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("https://mlproj.s3.amazonaws.com/input/used_car_data.csv")
df.dropna(inplace=True)

# Features and Target
X = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count']]
y = df['price_usd']

# Load your best model (e.g., Decision Tree or Random Forest)
model = joblib.load('car_price_model_rf.pkl')  # or change to _dt.pkl

# Predict
predictions = model.predict(X)

# Create DataFrame of predictions vs actual
result_df = pd.DataFrame({
    'Actual Price': y.values,
    'Predicted Price': predictions,
    'Error': np.abs(y.values - predictions)
})

# Sort by largest errors
sorted_errors = result_df.sort_values(by='Error', ascending=False)

# Show top 10 errors
print("🔍 Top 10 Worst Predictions:\n")
print(sorted_errors.head(10))

# Optional: Save to CSV for report reference
sorted_errors.head(10).to_csv('worst_predictions.csv', index=False)