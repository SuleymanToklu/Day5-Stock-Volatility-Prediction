import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from xgboost import XGBRegressor
import joblib
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

load_dotenv()

def run_training_pipeline():
    print("--- Training Pipeline Started ---")

    print("1/4 - Downloading historical stock data for SPY...")
    
    API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    
    df, meta_data = ts.get_daily(symbol='SPY', outputsize='full')
    
    df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.loc['2010-01-01':'2023-12-31']

    print("2/4 - Engineering features...")
    df['Returns'] = df['Close'].pct_change()

    for i in range(1, 6):
        df[f'Lag_{i}'] = df['Returns'].shift(i)

    df['MA_10'] = df['Returns'].rolling(window=10).mean()
    df['Volatility_10'] = df['Returns'].rolling(window=10).std()
    df['Target_Volatility'] = df['Returns'].rolling(window=5).std().shift(-5)
    
    df.dropna(inplace=True)

    features = [f'Lag_{i}' for i in range(1, 6)] + ['MA_10', 'Volatility_10']
    X = df[features]
    y = df['Target_Volatility']
    
    print("3/4 - Training XGBoost Regressor model...")
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X, y)

    print("4/4 - Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    print("--- Training Pipeline Completed Successfully! ---")

if __name__ == "__main__":
    run_training_pipeline()