import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load and clean dataset
df = pd.read_csv("https://mlproj.s3.amazonaws.com/input/used_car_data.csv")
df.dropna(inplace=True)

# Select features
X = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count']]
y = df['price_usd']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'car_price_model_lr.pkl')
print("✅ Linear Regression model saved as car_price_model_lr.pkl")