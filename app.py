from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
import pandas as pd
import numpy as np
import os
import math
import joblib
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
import logging
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')

# Global variables
model = None
scaler = None
total_profit = 0
total_trades = 0
price_history = deque(maxlen=1000)
prediction_history = deque(maxlen=1000)
trade_history = deque(maxlen=1000)
auto_retrain = True  # Auto retrain is enabled by default

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler if they exist
def load_existing_model():
    global model, scaler
    if os.path.exists('eth_price_model.h5') and os.path.exists('scaler.save'):
        try:
            model = load_model('eth_price_model.h5')
            scaler = joblib.load('scaler.save')
            logger.info("Model and scaler loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None
            scaler = None
    else:
        # Initialize model and scaler if not loaded
        model = None
        scaler = None

load_existing_model()

# Binance client
def get_client():
    # Use environment variables or secure storage for API keys
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    if not api_key or not api_secret:
        return None
    return Client(api_key, api_secret)

def get_eth_data(limit=1000, interval=Client.KLINE_INTERVAL_1MINUTE):
    client = get_client()
    if client is None:
        logger.error("Binance client not available. Please check API keys.")
        return None
    try:
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
    except Exception as e:
        logger.error(f"Error fetching ETH data: {e}")
        return None

def add_technical_indicators(df):
    # Existing indicators
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['RSI_14'] = compute_rsi(df['close'], 14)

    # New indicators
    df['MACD'] = compute_macd(df['close'])
    df['MACD_Signal'] = compute_macd_signal(df['close'])
    df['Bollinger_Upper'], df['Bollinger_Lower'] = compute_bollinger_bands(df['close'])
    df['Stochastic_Oscillator'] = compute_stochastic_oscillator(df['close'], df['low'], df['high'])

    df.fillna(method='backfill', inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    return RSI

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    return macd

def compute_macd_signal(series):
    macd = compute_macd(series)
    signal = macd.ewm(span=9, adjust=False).mean()
    return signal

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_stochastic_oscillator(close, low, high, period=14):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    stochastic = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    return stochastic

def preprocess_data(df):
    global scaler
    # Using MinMaxScaler for data normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data

def create_input_sequence(data, look_back=60):
    last_sequence = data[-look_back:]
    return np.array([last_sequence])

def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i, 3])  # 'close' price index
    return np.array(X), np.array(y)

def create_improved_model(input_shape):
    from tensorflow.keras.optimizers import Adam
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(GRU(units=64))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.00005)  # Lower learning rate
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def train_model(df):
    global model
    try:
        df = add_technical_indicators(df)
        scaled_data = preprocess_data(df)
        look_back = 60
        X, y = create_dataset(scaled_data, look_back)
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        model = create_improved_model((X_train.shape[1], X_train.shape[2]))

        # Implement Early Stopping and Learning Rate Reduction
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

        # Increase epochs and adjust batch size
        history = model.fit(
            X_train, y_train,
            epochs=100,  # Increased number of epochs
            batch_size=128,  # Adjusted batch size
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )

        # Load the best model
        model.load_weights('best_model.h5')

        # Evaluate model performance
        y_pred = model.predict(X_val)
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        logger.info(f'Validation MSE: {mse}')
        logger.info(f'Validation MAE: {mae}')
        logger.info(f'Validation MAPE: {mape}')
        logger.info(f'Validation RMSE: {rmse}')

        return True
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return False

def generate_future_predictions(model, last_sequence, future_steps=10):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(future_steps):
        prediction = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        predictions.append(prediction[0][0])
        # Update the sequence by appending the prediction and removing the oldest value
        new_row = np.append(current_sequence[-1][1:], prediction[0][0])
        current_sequence = np.vstack([current_sequence[1:], new_row])
    return predictions

def check_trading_condition(predictions, current_price):
    """
    Checks if the LSTM predictions indicate a 2% increase or decrease from the current price.
    Returns 'buy', 'sell', or 'hold'.
    """
    max_predicted_price = max(predictions)
    min_predicted_price = min(predictions)

    increase_percentage = ((max_predicted_price - current_price) / current_price) * 100
    decrease_percentage = ((current_price - min_predicted_price) / current_price) * 100

    if increase_percentage >= 2:
        return 'buy'
    elif decrease_percentage >= 2:
        return 'sell'
    else:
        return 'hold'

def execute_trade(action, current_price):
    """
    Executes a trade (buy or sell) on Binance.
    """
    global total_profit, total_trades
    client = get_client()
    if client is None:
        logger.error("Client is not available. Cannot execute trade.")
        return

    symbol = 'ETHUSDT'
    quantity = get_trade_quantity(client, symbol)

    try:
        if action == 'buy':
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(f"Buy order executed: {order}")
            # Update total trades
            total_trades += 1
            # Store the buy price for profit calculation
            session['last_buy_price'] = current_price
        elif action == 'sell':
            order = client.create_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(f"Sell order executed: {order}")
            # Update total trades
            total_trades += 1
            # Calculate profit
            last_buy_price = session.get('last_buy_price')
            if last_buy_price:
                profit = (current_price - last_buy_price) * quantity
                total_profit += profit
                session.pop('last_buy_price', None)
    except Exception as e:
        logger.error(f"Error executing {action} order: {e}")

def get_trade_quantity(client, symbol):
    """
    Calculates the trade quantity based on available balance and symbol specifications.
    """
    # Get account balance
    balance = client.get_asset_balance(asset='USDT')
    usdt_balance = float(balance['free'])
    # For simplicity, we'll use 10% of available USDT balance per trade
    trade_amount_usdt = usdt_balance * 0.10

    # Get symbol info to determine minimum quantity and step size
    symbol_info = client.get_symbol_info(symbol)
    step_size = None
    min_qty = None

    for filt in symbol_info['filters']:
        if filt['filterType'] == 'LOT_SIZE':
            step_size = float(filt['stepSize'])
            min_qty = float(filt['minQty'])
            break

    # Get current price
    ticker = client.get_symbol_ticker(symbol=symbol)
    current_price = float(ticker['price'])

    # Calculate quantity
    quantity = trade_amount_usdt / current_price

    # Adjust quantity to step size
    precision = int(round(-math.log(step_size, 10), 0))
    quantity = round(quantity, precision)

    # Ensure quantity is above minimum quantity
    if quantity < min_qty:
        quantity = min_qty

    return quantity

def automated_trading_task():
    with app.app_context():
        global total_profit, total_trades, price_history, prediction_history, trade_history
        logger.info("Running automated trading task.")
        try:
            client = get_client()
            if client is None:
                logger.error("Client is not available. Cannot execute trading task.")
                return

            eth_data = get_eth_data()
            if eth_data is None:
                logger.error("Error fetching ETH data. Please check your API keys.")
                return

            eth_data = add_technical_indicators(eth_data)
            eth_data_scaled = preprocess_data(eth_data)

            last_sequence = eth_data_scaled[-60:]

            # Check if model input shape matches data input shape
            expected_input_shape = model.input_shape  # Should be (None, look_back, num_features)
            actual_input_shape = (1, last_sequence.shape[0], last_sequence.shape[1])  # Shape for prediction

            if expected_input_shape[1:] != actual_input_shape[1:]:
                # Input shapes do not match; retrain the model
                logger.warning("Model input shape does not match data shape. Retraining the model.")
                success = train_model(eth_data)
                if success:
                    model.save('eth_price_model.h5')
                    joblib.dump(scaler, 'scaler.save')
                    logger.info("Model retrained and saved successfully.")
                    # Reload the model
                    load_existing_model()
                    # Recreate scaled data and last sequence
                    eth_data_scaled = preprocess_data(eth_data)
                    last_sequence = eth_data_scaled[-60:]
                else:
                    logger.error("Model retraining failed.")
                    return

            predictions_scaled = generate_future_predictions(model, last_sequence, future_steps=10)

            # Inverse transform predictions
            predictions = []
            for pred_scaled in predictions_scaled:
                last_scaled_data_point = eth_data_scaled[-1, :].copy()
                predicted_scaled_data_point = last_scaled_data_point.copy()
                predicted_scaled_data_point[3] = pred_scaled  # Replace 'close' price (index 3)
                predicted_scaled_data_point = predicted_scaled_data_point.reshape(1, -1)
                pred_inverse = scaler.inverse_transform(predicted_scaled_data_point)[0][3]
                predictions.append(float(pred_inverse))

            # Automatic trading logic
            current_price = eth_data['close'].iloc[-1]
            action = check_trading_condition(predictions, current_price)
            if action in ['buy', 'sell']:
                execute_trade(action, current_price)
                trade_history.append(action)
                logger.info(f"Executed {action} action.")
            else:
                trade_history.append('hold')
                logger.info("Holding position.")

            # Update price and prediction histories
            timestamp = eth_data.index[-1]
            price_history.append({'timestamp': timestamp, 'price': current_price})
            prediction_history.append({'timestamp': timestamp, 'prediction': predictions[0]})
            logger.info("Updated price and prediction histories.")

        except Exception as e:
            logger.error(f"Error in automated_trading_task: {e}")

def retrain_model_task():
    with app.app_context():
        global model
        logger.info("Starting model retraining task.")

        # Fetch fresh data
        eth_data = get_eth_data(limit=1000)
        if eth_data is None:
            logger.error("Error fetching ETH data for retraining.")
            return

        # Retrain the model
        success = train_model(eth_data)
        if success:
            model.save('eth_price_model.h5')
            joblib.dump(scaler, 'scaler.save')
            logger.info("Model retrained and saved successfully.")
        else:
            logger.error("Model retraining failed.")

# Scheduler setup
scheduler = BackgroundScheduler()
# Schedule automated trading task every 1 minute
scheduler.add_job(func=automated_trading_task, trigger="interval", minutes=1)
# Schedule retrain every 6 hours
scheduler.add_job(func=retrain_model_task, trigger="interval", hours=6, id='auto_retrain_job')
auto_retrain = True

# Helper function for login required
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    if 'logged_in' not in session or not session['logged_in']:
        return redirect(url_for('login'))
    else:
        return redirect(url_for('trading'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')  # Fetch email field from the form
        password = request.form.get('password')  # Fetch password field from the form

        # Dummy authentication logic (replace with actual authentication)
        if email and password:
            session['user_id'] = email  # Simple way to store the logged-in user's email
            session['logged_in'] = True
            return redirect(url_for('enter_api_keys'))
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')

@app.route('/create-account', methods=['GET', 'POST'])
def create_account():
    if request.method == 'POST':
        email = request.form.get('email')  # Fetch email field from the form
        password = request.form.get('password')  # Fetch password field from the form

        # Normally, you'd save the new user to a database here
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('create_account.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/enter-api-keys', methods=['GET', 'POST'])
@login_required
def enter_api_keys():
    if request.method == 'POST':
        api_key = request.form.get('api_key')
        api_secret = request.form.get('api_secret')
        # Store API keys in environment variables or secure storage
        os.environ['BINANCE_API_KEY'] = api_key
        os.environ['BINANCE_API_SECRET'] = api_secret
        return redirect(url_for('trading'))
    return render_template('enter_api_keys.html')

@app.route('/trading')
@login_required
def trading():
    global total_profit, total_trades, model, scaler

    if 'BINANCE_API_KEY' not in os.environ or 'BINANCE_API_SECRET' not in os.environ:
        return redirect(url_for('enter_api_keys'))

    if model is None or scaler is None:
        # Try loading the existing model again
        load_existing_model()
        if model is None or scaler is None:
            return render_template('error.html', message="Model is not loaded. Please train the model first.")

    eth_data = get_eth_data()
    if eth_data is None:
        return render_template('error.html', message="Error fetching ETH data. Please check your API keys.")

    eth_data = add_technical_indicators(eth_data)
    eth_data_scaled = preprocess_data(eth_data)

    # Create input sequence
    input_sequence = create_input_sequence(eth_data_scaled)
    expected_input_shape = model.input_shape  # Should be (None, look_back, num_features)
    actual_input_shape = input_sequence.shape  # Should be (1, look_back, num_features)

    if expected_input_shape[1:] != actual_input_shape[1:]:
        # Input shapes do not match; retrain the model
        logger.warning("Model input shape does not match data shape. Retraining the model.")
        success = train_model(eth_data)
        if success:
            model.save('eth_price_model.h5')
            joblib.dump(scaler, 'scaler.save')
            logger.info("Model retrained and saved successfully.")
            # Reload the model
            load_existing_model()
            # Recreate input sequence after retraining
            eth_data_scaled = preprocess_data(eth_data)
            input_sequence = create_input_sequence(eth_data_scaled)
        else:
            return render_template('error.html', message="Model retraining failed.")

    # Proceed with prediction
    prediction_scaled = model.predict(input_sequence)

    # Correct inverse transformation
    last_scaled_data_point = eth_data_scaled[-1, :].copy()
    predicted_scaled_data_point = last_scaled_data_point.copy()
    predicted_scaled_data_point[3] = prediction_scaled[0][0]  # Replace 'close' price (index 3)
    predicted_scaled_data_point = predicted_scaled_data_point.reshape(1, -1)

    next_price = scaler.inverse_transform(predicted_scaled_data_point)[0][3]  # 'close' price index

    # Calculate prediction change
    current_price = eth_data['close'].iloc[-1]
    prediction_change = ((next_price - current_price) / current_price) * 100

    return render_template('trading.html', next_price=next_price, total_profit=total_profit, total_trades=total_trades,
                           prediction_change=prediction_change, auto_retrain=auto_retrain)

@app.route('/get-historical-data')
@login_required
def get_historical_data():
    data_length = 100  # Limit to last 100 entries

    # Calculate prediction change
    if prediction_history and price_history:
        last_prediction = prediction_history[-1]['prediction']
        last_price = price_history[-1]['price']
        prediction_change = ((last_prediction - last_price) / last_price) * 100
    else:
        prediction_change = 0
        last_prediction = 0

    data = {
        'timestamps': [p['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for p in list(price_history)[-data_length:]],
        'actual_prices': [p['price'] for p in list(price_history)[-data_length:]],
        'predicted_prices': [p['prediction'] for p in list(prediction_history)[-data_length:]],
        'trade_actions': list(trade_history)[-data_length:],
        'metrics': {
            'next_price': last_prediction,
            'prediction_change': prediction_change,
            'total_profit': total_profit,
            'total_trades': total_trades
        }
    }
    return jsonify(data)

@app.route('/get-predictions')
@login_required
def get_predictions():
    # Generate future predictions
    eth_data = get_eth_data()
    if eth_data is None:
        return jsonify({'predictions': []})
    eth_data = add_technical_indicators(eth_data)
    eth_data_scaled = preprocess_data(eth_data)
    last_sequence = eth_data_scaled[-60:]

    # Check if model input shape matches data input shape
    expected_input_shape = model.input_shape
    actual_input_shape = (1, last_sequence.shape[0], last_sequence.shape[1])

    if expected_input_shape[1:] != actual_input_shape[1:]:
        # Input shapes do not match; retrain the model
        logger.warning("Model input shape does not match data shape. Retraining the model.")
        success = train_model(eth_data)
        if success:
            model.save('eth_price_model.h5')
            joblib.dump(scaler, 'scaler.save')
            logger.info("Model retrained and saved successfully.")
            # Reload the model
            load_existing_model()
            # Recreate scaled data and last sequence
            eth_data_scaled = preprocess_data(eth_data)
            last_sequence = eth_data_scaled[-60:]
        else:
            logger.error("Model retraining failed.")
            return jsonify({'predictions': []})

    predictions_scaled = generate_future_predictions(model, last_sequence, future_steps=10)
    predictions = []
    for pred_scaled in predictions_scaled:
        last_scaled_data_point = eth_data_scaled[-1, :].copy()
        predicted_scaled_data_point = last_scaled_data_point.copy()
        predicted_scaled_data_point[3] = pred_scaled  # Replace 'close' price (index 3)
        predicted_scaled_data_point = predicted_scaled_data_point.reshape(1, -1)
        pred_inverse = scaler.inverse_transform(predicted_scaled_data_point)[0][3]
        predictions.append(float(pred_inverse))
    return jsonify({'predictions': predictions})

@app.route('/manual-buy')
@login_required
def manual_buy():
    current_price = get_eth_data()['close'].iloc[-1]
    execute_trade('buy', current_price=current_price)
    return jsonify({'message': 'Manual buy action executed successfully'})

@app.route('/manual-sell')
@login_required
def manual_sell():
    current_price = get_eth_data()['close'].iloc[-1]
    execute_trade('sell', current_price=current_price)
    return jsonify({'message': 'Manual sell action executed successfully'})

@app.route('/start-retrain', methods=['POST'])
@login_required
def start_retrain():
    global model
    eth_data = get_eth_data(limit=1000)
    if eth_data is None:
        return jsonify({'message': 'Error fetching ETH data for retraining.'}), 500

    success = train_model(eth_data)
    if success:
        model.save('eth_price_model.h5')
        joblib.dump(scaler, 'scaler.save')
        return jsonify({'message': 'Model retrained successfully!'})
    else:
        return jsonify({'message': 'Model retraining failed.'}), 500

@app.route('/toggle-auto-retrain', methods=['POST'])
@login_required
def toggle_auto_retrain():
    global auto_retrain
    if auto_retrain:
        # Stop auto retrain
        scheduler.remove_job('auto_retrain_job')
        auto_retrain = False
        message = 'Automatic retraining stopped.'
    else:
        # Start auto retrain
        scheduler.add_job(func=retrain_model_task, trigger="interval", hours=6, id='auto_retrain_job')
        auto_retrain = True
        message = 'Automatic retraining started.'
    return jsonify({'message': message})

if __name__ == '__main__':
    scheduler.start()
    app.run(debug=True)
