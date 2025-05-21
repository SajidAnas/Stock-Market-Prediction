import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

if __name__ == "__main__":
    # Choose a stock and time period
    stock_name = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    # Fetch the data
    stock_data = yf.download(stock_name, start=start_date, end=end_date, progress=False)
    
    # Prepare the data
    data = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences for training
    seq_length = 10
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train the model
    model = build_lstm_model(seq_length, 1)
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Plot the results
    plt.figure(figsize=(15, 7))
    plt.plot(stock_data.index[seq_length:train_size+seq_length], y_train_inv, label='Actual (Training)')
    plt.plot(stock_data.index[seq_length:train_size+seq_length], train_predictions, label='Predicted (Training)')
    plt.plot(stock_data.index[train_size+seq_length:], y_test_inv, label='Actual (Testing)')
    plt.plot(stock_data.index[train_size+seq_length:], test_predictions, label='Predicted (Testing)')
    plt.title(f'{stock_name} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Calculate and print the model's performance metrics
    train_rmse = np.sqrt(np.mean((train_predictions - y_train_inv) ** 2))
    test_rmse = np.sqrt(np.mean((test_predictions - y_test_inv) ** 2))
    print(f"\nTraining RMSE: ${train_rmse:.2f}")
    print(f"Testing RMSE: ${test_rmse:.2f}")