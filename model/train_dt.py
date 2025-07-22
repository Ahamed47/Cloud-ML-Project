import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load dataset
df = pd.read_csv("https://mlproj.s3.amazonaws.com/input/used_car_data.csv")
df.dropna(inplace=True)

# Select features and target
X = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count']]
y = df['price_usd']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Decision Tree
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'car_price_model_dt.pkl')
print("✅ Decision Tree model saved as car_price_model_dt.pkl")