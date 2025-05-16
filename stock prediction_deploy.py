# app.py
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("Tesla Stock Price Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload Tesla Stock CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df['previous_close'] = df['close'].shift(1)
    df['pct_change'] = (df['close'] - df['open']) / df['open']
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df.dropna(inplace=True)

    features = ['open', 'high', 'low', 'volume',
                'previous_close', 'pct_change', 'ma_5', 'ma_10',
                'day_of_week', 'month']
    target = 'close'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"])

    if model == "Linear Regression":
        reg = LinearRegression()
    else:
        reg = RandomForestRegressor(n_estimators=100, random_state=42)

    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)

    st.subheader("Model Evaluation")
    st.write("MAE:", mean_absolute_error(y_test, preds))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    st.write("R² Score:", r2_score(y_test, preds))

    st.subheader("Actual vs Predicted Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['date'].iloc[-len(y_test):], y_test.values, label='Actual', color='black')
    ax.plot(df['date'].iloc[-len(y_test):], preds, label='Predicted', linestyle='--')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Tesla Stock Price Prediction")
    ax.legend()
    st.pyplot(fig)
