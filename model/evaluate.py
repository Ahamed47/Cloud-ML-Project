import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("https://mlproj.s3.amazonaws.com/input/used_car_data.csv")
df.dropna(inplace=True)

# Features and Target
X = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count']]
y = df['price_usd']

# Load models
models = {
    "Random Forest": joblib.load('car_price_model_rf.pkl'),
    "Linear Regression": joblib.load('car_price_model_lr.pkl'),
    "Decision Tree": joblib.load('car_price_model_dt.pkl'),
}

# Evaluation function
def evaluate_model(name, model):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"\n📊 {name} Performance:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

# Evaluate all
for name, model in models.items():
    evaluate_model(name, model)