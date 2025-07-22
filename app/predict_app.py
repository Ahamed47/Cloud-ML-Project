import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from io import BytesIO

st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
st.title("🚗 Used Car Price Predictor")

st.markdown("Select a model and enter car details below to estimate the price of a used car.")

# 🔍 Model selection
model_choice = st.selectbox("Choose Prediction Model", [
    "Random Forest",
    "Linear Regression",
    "Decision Tree"
])

# Map model names to file paths
model_files = {
    "Random Forest": "../model/car_price_model_rf.pkl",
    "Linear Regression": "../model/car_price_model_lr.pkl",
    "Decision Tree": "../model/car_price_model_dt.pkl"
}

# Load selected model
model = joblib.load(model_files[model_choice])

# 🔢 Input fields
make_year = st.number_input("Car Make Year", min_value=1990, max_value=2024, value=2015)
mileage_kmpl = st.number_input("Mileage (in KMPL)", min_value=0.0, step=0.1)
engine_cc = st.number_input("Engine Size (in CC)", min_value=500, step=100)
owner_count = st.selectbox("Number of Previous Owners", [0, 1, 2, 3, 4])

# 🧠 Predict
if st.button("Predict Price"):
    input_data = np.array([[make_year, mileage_kmpl, engine_cc, owner_count]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Estimated Price: ${predicted_price:,.2f} USD")

# 📊 Error Analysis Section
st.markdown("---")
st.subheader("🔎 Error Analysis (Top 10 Wrong Predictions)")

if st.checkbox("Show error analysis table and chart"):
    # Load dataset
    df = pd.read_csv("https://mlproj.s3.amazonaws.com/input/used_car_data.csv")
    df.dropna(inplace=True)

    X = df[['make_year', 'mileage_kmpl', 'engine_cc', 'owner_count']]
    y = df['price_usd']
    predictions = model.predict(X)

    # Create results DataFrame
    result_df = pd.DataFrame({
        'Actual Price': y.values,
        'Predicted Price': predictions,
        'Error': np.abs(y.values - predictions)
    })

    sorted_errors = result_df.sort_values(by='Error', ascending=False).head(10)
    st.dataframe(sorted_errors)

    # Plot chart
    st.subheader("📉 Bar Chart: Actual vs Predicted Prices (Top 10 Errors)")
    fig, ax = plt.subplots()
    sorted_errors[['Actual Price', 'Predicted Price']].plot(kind='bar', ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Price (USD)")
    plt.title("Actual vs Predicted (Top 10 Errors)")
    st.pyplot(fig)
    image_buffer = BytesIO()
    plt.savefig(image_buffer, format='png')
    image_buffer.seek(0)

    # Upload to S3
    s3 = boto3.client("s3")
    s3.upload_fileobj(
        image_buffer,
        Bucket="mlproj",
        Key="outputs/top_10_error_chart.png"
    )

    st.success("📤 Chart uploaded to s3://mlproj/outputs/top_10_error_chart.png")

st.markdown("---")
st.subheader("⚙️ Model Performance Summary")

if st.checkbox("Show model training time and latency table"):
    try:
        perf_df = pd.read_csv('../model/performance_summary.csv')
        st.dataframe(perf_df)

        st.success("✅ This table compares training speed and latency across models.")
    except FileNotFoundError:
        st.error("performance_summary.csv not found. Please run performance_analysis.py first.")