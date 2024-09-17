# train_model.py

import pandas as pd
import numpy as np
from binance.client import Client
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib  # For saving the scaler
import os

# Load Binance API keys from environment variables
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')
client = Client(API_KEY, API_SECRET)

def get_eth_data(interval='1h', limit=1000):
    klines = client.get_klines(symbol='ETHUSDT', interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.astype(float)
    return df

def add_technical_indicators(df):
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['RSI_14'] = compute_rsi(df['close'], 14)
    df.fillna(method='backfill', inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 3])  # 'close' price index
    return np.array(X), np.array(y)

def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model():
    df = get_eth_data()
    df = add_technical_indicators(df)
    scaled_data, scaler = preprocess_data(df)
    look_back = 60
    X, y = create_dataset(scaled_data, look_back)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = create_model((X_train.shape[1], X_train.shape[2]))
    
    # Implement Early Stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, 
              validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Evaluate model performance
    y_pred = model.predict(X_val)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_val, y_pred)
    print(f'Validation MSE: {mse}')

    # Save the model and scaler
    model.save('eth_price_model.h5')
    joblib.dump(scaler, 'scaler.save')

if __name__ == '__main__':
    train_model()
