import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import boto3
from io import StringIO

# Load and clean dataset
df = pd.read_csv("https://mlproj.s3.amazonaws.com/input/used_car_data.csv")
df.dropna(inplace=True)
X = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count']]
y = df['price_usd']

# Define models
models = {
    "Random Forest": RandomForestRegressor(),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor()
}

# Create result table
performance = []

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔧 {name} Model:")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Training Time
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()
    train_duration = end_train - start_train

    # Prediction Latency
    sample = X_test.iloc[[0]]
    start_pred = time.time()
    model.predict(sample)
    end_pred = time.time()
    latency = end_pred - start_pred

    performance.append((name, round(train_duration, 4), round(latency, 6)))

    # Learning Curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring='neg_mean_squared_error'
    )
    train_errors = -train_scores.mean(axis=1)
    test_errors = -test_scores.mean(axis=1)

    plt.plot(train_sizes, train_errors, label=f'{name} Train')
    plt.plot(train_sizes, test_errors, label=f'{name} Test')

# Format plot
plt.title("Learning Curves")
plt.xlabel("Training Set Size")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("learning_curves.png")
plt.show()

# Print and save performance table
print("\n📋 Model Performance Table:")
print("{:<20} {:<20} {:<20}".format("Model", "Training Time (s)", "Latency (s)"))
for row in performance:
    print("{:<20} {:<20} {:<20}".format(*row))

df_perf = pd.DataFrame(performance, columns=["Model", "Training Time (s)", "Latency (s)"])
df_perf.to_csv("performance_summary.csv", index=False)


# Save to memory buffer
csv_buffer = StringIO()
df_perf.to_csv(csv_buffer, index=False)

# Upload to S3
s3 = boto3.client("s3")
bucket_name = "mlproj"
s3_key = "outputs/performance_summary.csv"

# Put the object to S3 (private)
s3.put_object(
    Bucket=bucket_name,
    Key=s3_key,
    Body=csv_buffer.getvalue()
)

print(f"\n✅ CSV uploaded to s3://{bucket_name}/{s3_key}")