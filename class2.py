# class2.py

# Importaciones necesarias
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Para indicadores técnicos
import ta

# Para escalado y división de datos
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Para construir y entrenar el modelo LSTM
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

# Para Early Stopping
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Para guardar y cargar el scaler
import joblib

# Para métricas y evaluación
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# Para pruebas de estacionariedad y descomposición
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Desactivar advertencias si es necesario
import warnings
warnings.filterwarnings('ignore')

#______________________________________________________________________________________
##### Definición de Clases

class DataLoader:
    def __init__(self, symbol="SHIB/USDT", timeframe='15m', since_days=180):
        self.symbol = symbol
        self.timeframe = timeframe
        self.since_days = since_days
        self.data = None

    def fetch_data(self):
        exchange = ccxt.binance()
        data = []
        since = exchange.parse8601(
            (datetime.utcnow() - timedelta(days=self.since_days)).strftime('%Y-%m-%dT%H:%M:%SZ')
        )

        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    self.symbol, timeframe=self.timeframe, since=since, limit=1000
                )
                if not ohlcv:
                    break
                data += ohlcv
                since = ohlcv[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)
            except Exception as e:
                print("Error:", e)
                break

        df = pd.DataFrame(
            data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        self.data = df
        return self.data

    def save_data(self, filename):
        if self.data is not None:
            self.data.to_csv(filename)
        else:
            print("No hay datos para guardar.")

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
        self.scaler = None
        self.sequence_length = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def add_features(self):
        df = self.data.copy()
        # Indicadores técnicos
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['MACD'] = ta.trend.MACD(df['close']).macd()
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['Bollinger_High'] = bollinger.bollinger_hband()
        df['Bollinger_Low'] = bollinger.bollinger_lband()
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        df['Rolling_STD'] = df['close'].rolling(window=20).std()
        df['Volume_EMA'] = df['volume'].ewm(span=10, adjust=False).mean()
        df = df.dropna()
        self.data = df
        return self.data

    def check_stationarity(self, series):
        # Prueba de Dickey-Fuller ADF
        result_adf = adfuller(series.dropna())
        print('ADF Statistic: %f' % result_adf[0])
        print('p-value: %f' % result_adf[1])
        for key, value in result_adf[4].items():
            print(f'Critical Values {key}: {value}')

        # Prueba KPSS
        kpss_stat, p_value, lags, crit_values = kpss(series.dropna())
        print(f'KPSS Statistic: {kpss_stat}')
        print(f'p-value: {p_value}')
        print('Critical Values:')
        for key, value in crit_values.items():
            print(f'   {key}: {value}')

    def difference_series(self, series):
        diff_series = series.diff().dropna()
        return diff_series

    def plot_decomposition(self, series, model='additive', freq=None):
        decomposition = seasonal_decompose(series, model=model, period=freq)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(14, 10))
        plt.subplot(411)
        plt.plot(series, label='Original')
        plt.legend(loc='upper left')
        plt.title('Descomposición de la Serie Temporal')

        plt.subplot(412)
        plt.plot(trend, label='Tendencia')
        plt.legend(loc='upper left')

        plt.subplot(413)
        plt.plot(seasonal, label='Estacionalidad')
        plt.legend(loc='upper left')

        plt.subplot(414)
        plt.plot(residual, label='Residuales')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()

    def plot_acf_pacf(self, series, lags=50):
        plt.figure(figsize=(16, 6))
        plt.subplot(121)
        plot_acf(series.dropna(), ax=plt.gca(), lags=lags)
        plt.subplot(122)
        plot_pacf(series.dropna(), ax=plt.gca(), lags=lags)
        plt.show()

    def scale_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.data)
        return scaled_data

    def create_sequences(self, scaled_data, sequence_length=60):
        self.sequence_length = sequence_length
        X = []
        y = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, self.data.columns.get_loc('close')])
        X, y = np.array(X), np.array(y)
        return X, y

    def prepare_data(self, test_size=0.2, sequence_length=60):
        self.add_features()
        # Opcional: verificar estacionariedad y diferenciar la serie
        # self.check_stationarity(self.data['close'])
        # self.data['close_diff'] = self.difference_series(self.data['close'])
        scaled_data = self.scale_data()
        X, y = self.create_sequences(scaled_data, sequence_length)
        split = int(len(X) * (1 - test_size))
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]
        return self.X_train, self.X_test, self.y_train, self.y_test

class LSTMModel:
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape

    def build_model(self):
        model = Sequential()
        model.add(
            LSTM(units=256, return_sequences=True, input_shape=self.input_shape)
        )
        model.add(Dropout(0.2))
        model.add(LSTM(units=256))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        # Compilar el modelo con métricas adicionales
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mape'])
        self.model = model
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            shuffle=False
        )
        return history

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = load_model(filename)

class TradingStrategy:
    def __init__(self, scaler, data_columns):
        self.scaler = scaler
        self.data_columns = data_columns

    def predict(self, model, X):
        predictions = model.predict(X)
        # Desescalar las predicciones
        predictions_full = np.zeros((len(predictions), len(self.data_columns)))
        predictions_full[:, self.data_columns.get_loc('close')] = predictions.flatten()
        predictions_descaled = self.scaler.inverse_transform(predictions_full)[:, self.data_columns.get_loc('close')]
        return predictions_descaled

    def generate_signals(self, predictions, y_real):
        signals = np.where(predictions > y_real.shift(1), 1, -1)
        signals[0] = 0  # Asignar 0 al primer valor
        return signals

    def backtest(self, y_real, signals):
        results = pd.DataFrame({
            'Real': y_real,
            'Señal': signals
        })
        results['Retorno'] = results['Real'].pct_change()
        results['Estrategia'] = results['Retorno'] * results['Señal'].shift(1)
        results['Retorno Acumulado'] = (1 + results['Retorno']).cumprod()
        results['Estrategia Acumulada'] = (1 + results['Estrategia']).cumprod()
        return results

    def calculate_sharpe_ratio(self, strategy_returns):
        sharpe_ratio = (
            strategy_returns.mean() / strategy_returns.std()
        ) * np.sqrt(252 * 24 * 4)
        return sharpe_ratio

class Visualizer:
    def __init__(self):
        pass

    def plot_predictions(self, y_real, predictions):
        plt.figure(figsize=(14, 7))
        plt.plot(y_real.index, y_real.values, label='Precio Real')
        plt.plot(y_real.index, predictions, label='Predicción')
        plt.title('Precio Real vs Predicción')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre')
        plt.legend()
        plt.show()

    def plot_signals(self, y_real, predictions, signals):
        plt.figure(figsize=(14, 7))
        plt.plot(y_real.index, y_real.values, label='Precio Real')
        plt.plot(
            y_real.index[signals == 1],
            y_real[signals == 1],
            '^',
            markersize=10,
            color='g',
            label='Comprar'
        )
        plt.plot(
            y_real.index[signals == -1],
            y_real[signals == -1],
            'v',
            markersize=10,
            color='r',
            label='Vender'
        )
        plt.title('Señales de Compra y Venta')
        plt.xlabel('Fecha')
        plt.ylabel('Precio de Cierre')
        plt.legend()
        plt.show()

    def plot_backtest(self, results):
        plt.figure(figsize=(14, 7))
        plt.plot(results.index, results['Retorno Acumulado'], label='Retorno Acumulado del Mercado')
        plt.plot(results.index, results['Estrategia Acumulada'], label='Retorno Acumulado de la Estrategia')
        plt.title('Comparación del Rendimiento')
        plt.xlabel('Fecha')
        plt.ylabel('Retorno Acumulado')
        plt.legend()
        plt.show()

    def plot_training_history(self, history):
        # Graficar la pérdida durante el entrenamiento
        plt.figure(figsize=(14, 5))
        plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
        plt.plot(history.history['val_loss'], label='Pérdida de validación')
        plt.title('Pérdida del modelo durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()

        # Graficar MAE durante el entrenamiento
        plt.figure(figsize=(14, 5))
        plt.plot(history.history['mae'], label='MAE de entrenamiento')
        plt.plot(history.history['val_mae'], label='MAE de validación')
        plt.title('MAE del modelo durante el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

class FuturePredictor:
    def __init__(self, scaler, data_columns, model, sequence_length=60):
        self.scaler = scaler
        self.data_columns = data_columns
        self.model = model
        self.sequence_length = sequence_length

    def fetch_recent_data(self, symbol="SHIB/USDT", timeframe='15m', limit=200):
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def prepare_recent_data(self, data):
        # Mismos pasos de ingeniería de características
        feature_engineer = FeatureEngineer(data)
        data = feature_engineer.add_features()
        scaled_data = self.scaler.transform(data)
        X_recent = []
        for i in range(self.sequence_length, len(scaled_data) + 1):
            X_recent.append(scaled_data[i - self.sequence_length:i])
        X_recent = np.array(X_recent)
        return X_recent, data.index[self.sequence_length - 1:], data

    def predict_future(self, symbol="SHIB/USDT", timeframe='15m', limit=200):
        data = self.fetch_recent_data(symbol, timeframe, limit)
        X_recent, dates, data_with_features = self.prepare_recent_data(data)
        predictions = self.model.predict(X_recent)
        # Desescalar las predicciones
        predictions_full = np.zeros((len(predictions), len(self.data_columns)))
        predictions_full[:, self.data_columns.get_loc('close')] = predictions.flatten()
        predictions_descaled = self.scaler.inverse_transform(predictions_full)[:, self.data_columns.get_loc('close')]
        # Crear DataFrame de resultados
        results_future = pd.DataFrame({
            'Fecha': dates,
            'Prediccion': predictions_descaled
        })
        results_future.set_index('Fecha', inplace=True)
        return results_future, data_with_features
