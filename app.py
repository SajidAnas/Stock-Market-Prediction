import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
import requests

# Enable caching for yfinance data

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol, start_date, end_date):
    try:
        # Single API call to download data without custom session
        stock_data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if len(stock_data) == 0:
            # Try to get ticker info to verify if symbol exists
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info:
                return None, f"No data found for symbol '{symbol}'. Please verify the symbol is correct."
            return None, f"Found symbol '{symbol}' but no historical data available for the specified date range."
            
        return stock_data, None
        
    except Exception as e:
        error_msg = f"Error fetching data for {symbol}: {str(e)}"
        st.error(error_msg)
        return None, error_msg

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(seq_length, n_features):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, n_features), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def main():
    st.title("Stock Price Prediction using LSTM")
    st.write("This application predicts stock prices using LSTM neural networks.")

    # User input for stock symbol
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOGL, MSFT):", "AAPL").upper()
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3650)  # 10 years of data
    
    if st.button("Predict Stock Price"):
        try:
            # Fetch the data
            with st.spinner("Fetching stock data..."):
                st.write(f"Attempting to fetch data for {stock_symbol}")
                st.write(f"Date range: {start_date.date()} to {end_date.date()}")
                
                stock_data, error = fetch_stock_data(stock_symbol, start_date, end_date)
                if error:
                    st.error(error)
                    return
                
                st.write(f"Successfully downloaded data shape: {stock_data.shape}")
            
            # Prepare the data
            data = stock_data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences for training
            seq_length = 10
            X, y = create_sequences(scaled_data, seq_length)
            
            # Split the data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Build and train the model
            with st.spinner("Training the model..."):
                model = build_lstm_model(seq_length, 1)
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1,
                    verbose=0
                )
            
            # Make predictions
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            
            # Inverse transform predictions
            train_predictions = scaler.inverse_transform(train_predictions)
            test_predictions = scaler.inverse_transform(test_predictions)
            y_train_inv = scaler.inverse_transform(y_train)
            y_test_inv = scaler.inverse_transform(y_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(np.mean((train_predictions - y_train_inv) ** 2))
            test_rmse = np.sqrt(np.mean((test_predictions - y_test_inv) ** 2))
            
            # Display results
            st.subheader("Prediction Results")
            st.write(f"Training RMSE: ${train_rmse:.2f}")
            st.write(f"Testing RMSE: ${test_rmse:.2f}")
            
            # Plot the results
            fig, ax = plt.subplots(figsize=(15, 7))
            ax.plot(stock_data.index[seq_length:train_size+seq_length], y_train_inv, label='Actual (Training)')
            ax.plot(stock_data.index[seq_length:train_size+seq_length], train_predictions, label='Predicted (Training)')
            ax.plot(stock_data.index[train_size+seq_length:], y_test_inv, label='Actual (Testing)')
            ax.plot(stock_data.index[train_size+seq_length:], test_predictions, label='Predicted (Testing)')
            ax.set_title(f'{stock_symbol} Stock Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)
            
            # Display recent data
            st.subheader("Recent Stock Data")
            st.dataframe(stock_data.tail())
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Debug information:")
            st.write(f"Stock symbol: {stock_symbol}")
            st.write(f"Start date: {start_date}")
            st.write(f"End date: {end_date}")

if __name__ == "__main__":
    main() 