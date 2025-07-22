import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("https://mlproj.s3.amazonaws.com/input/used_car_data.csv")
df.dropna(inplace=True)

X = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count']]
y = df['price_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# ✅ Save with correct filename
joblib.dump(model, 'car_price_model_rf.pkl')
print("✅ Random Forest model saved as car_price_model_rf.pkl")