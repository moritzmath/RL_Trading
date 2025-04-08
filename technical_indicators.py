import pandas as pd
import numpy as np


def technical_indicators(df, sma_period=10, ema_period=14, crossover_short=5, crossover_long=20, rsi_period=14, macd_short=12, macd_long=26, macd_signal=9, bb_period=20, bb_std=2, so_period=14):
    """
    Adds common technical indicators and their corresponding trading signals to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'Close' column (and 'Open' for trading env use).
        sma_period (int): Window for Simple Moving Average.
        ema_period (int): Window for Exponential Moving Average.
        crossover_short (int): Short window for crossover strategy.
        crossover_long (int): Long window for crossover strategy.
        rsi_period (int): Period for Relative Strength Index.
        macd_short (int): Short-term EMA period for MACD.
        macd_long (int): Long-term EMA period for MACD.
        macd_signal (int): Signal line EMA period for MACD.
        bb_period (int): Window for Bollinger Bands.
        bb_std (int): Standard deviation multiplier for Bollinger Bands.
        so_period (int): Lookback period for Stochastic Oscillator.
    """
     
    # Simple Moving Average
    df['SMA'] = df['Close'].rolling(window=sma_period).mean()

    # Exponential Moving average
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    # Moving Average Crossover
    df['SMA_short'] = df['Close'].rolling(window=crossover_short).mean()
    df['SMA_long'] = df['Close'].rolling(window=crossover_long).mean()
    df['Crossover_signal'] = 0
    df.loc[(df['SMA_short'] > df['SMA_long']) & (df['SMA_short'].shift(1) <= df['SMA_long'].shift(1)), 'Crossover_signal'] = 1
    df.loc[(df['SMA_short'] < df['SMA_long']) & (df['SMA_short'].shift(1) >= df['SMA_long'].shift(1)), 'Crossover_signal'] = -1


    # Relative Strength Index
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain, avg_loss = gain.rolling(window=rsi_period).mean(), loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_signal = np.where(rsi > 70, 1, np.where(rsi < 30, -1, 0))
    df['RSI'] = rsi
    df['RSI_signal'] = rsi_signal

    # Moving Average Convergence Divergence
    ema_short = df['Close'].ewm(span=macd_short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=macd_long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
    df['MACD_signal'] = 0
    df.loc[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1)), 'MACD_signal'] = 1
    df.loc[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1)), 'MACD_signal'] = -1

    # Bollinger Bands
    middle_band = df['Close'].rolling(window=bb_period).mean()
    std = df['Close'].rolling(window=bb_period).std()
    df['Upper_Band'] = middle_band + (bb_std * std)
    df['Lower_Band'] = middle_band - (bb_std * std)
    df['BB_signal'] = 0
    df.loc[df['Close'] > df['Upper_Band'], 'BB_signal'] = -1  
    df.loc[df['Close'] < df['Lower_Band'], 'BB_signal'] = 1 

    # Stochastic Oscillator
    low_min = df['Close'].rolling(window=so_period).min()
    high_max = df['Close'].rolling(window=so_period).max()
    df['Stochastic_Osc'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['SO_signal'] = 0
    df.loc[df['Stochastic_Osc'] > 80, 'SO_signal'] = -1  
    df.loc[df['Stochastic_Osc'] < 20, 'SO_signal'] = 1 

    # Further Possible Indicators: Average True Range