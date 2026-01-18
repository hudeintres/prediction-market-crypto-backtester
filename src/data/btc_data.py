"""
BTC Price History Data Downloader

Downloads historical Bitcoin price data from Yahoo Finance API.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import json


class BTCDataDownloader:
    """
    Downloads and processes BTC price history data from Yahoo Finance.
    """
    
    YAHOO_SYMBOL = "BTC-USD"
    
    def __init__(self, data_dir: str = "data", use_synthetic: bool = False):
        """
        Initialize the BTC data downloader.
        
        Args:
            data_dir: Directory to store downloaded data
            use_synthetic: If True, generate synthetic data when real data unavailable
        """
        self.data_dir = data_dir
        self.use_synthetic = use_synthetic
        os.makedirs(data_dir, exist_ok=True)
        
    def download_price_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download BTC price history from Yahoo Finance.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (default: 2 years ago)
            end_date: End date in YYYY-MM-DD format (default: today)
            interval: Data interval ('1d', '1h', '5m', etc.)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            RuntimeError: If real data unavailable and use_synthetic is False
        """
        import yfinance as yf
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        
        print(f"Downloading BTC price history from {start_date} to {end_date}...")
        
        try:
            btc = yf.Ticker(self.YAHOO_SYMBOL)
            df = btc.history(start=start_date, end=end_date, interval=interval)
        except Exception as e:
            print(f"Error downloading from Yahoo Finance: {e}")
            df = pd.DataFrame()
        
        if df.empty:
            if self.use_synthetic:
                print("Yahoo Finance unavailable, generating synthetic BTC data...")
                df = self._generate_synthetic_price_data(start_date, end_date)
            else:
                raise RuntimeError(
                    f"Failed to download BTC price data from Yahoo Finance. "
                    f"Please check your internet connection or use --synthetic flag "
                    f"to generate synthetic data for testing."
                )
        else:
            print(f"Successfully downloaded {len(df)} price records from Yahoo Finance")
        
        # Clean up the dataframe
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime'])
            
        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, f"btc_price_history_{interval}.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved BTC price history to {filepath}")
        
        return df
    
    def _generate_synthetic_price_data(
        self,
        start_date: str,
        end_date: str,
        initial_price: float = 40000.0
    ) -> pd.DataFrame:
        """
        Generate synthetic BTC price data using geometric Brownian motion
        for testing when real data is unavailable.
        
        Args:
            start_date: Start date
            end_date: End date
            initial_price: Starting price
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        print("[SYNTHETIC DATA] Generating simulated BTC prices...")
        np.random.seed(42)  # For reproducibility
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        n_days = len(dates)
        
        # GBM parameters calibrated to BTC historical volatility
        mu = 0.0005  # Daily drift (annualized ~18%)
        sigma = 0.04  # Daily volatility (annualized ~76%)
        
        # Generate log returns
        dt = 1/365
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_days)
        
        # Calculate prices
        log_prices = np.log(initial_price) + np.cumsum(log_returns)
        close_prices = np.exp(log_prices)
        
        # Generate OHLCV
        daily_range = 0.03  # 3% average daily range
        high_prices = close_prices * (1 + daily_range * np.random.rand(n_days))
        low_prices = close_prices * (1 - daily_range * np.random.rand(n_days))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = initial_price
        volume = np.random.lognormal(mean=23, sigma=0.5, size=n_days)  # Typical BTC volume
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volume
        })
        
        return df
    
    def calculate_volatility(
        self,
        df: pd.DataFrame,
        window: int = 30,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate historical volatility from price data.
        
        Args:
            df: DataFrame with price data (must have 'close' column)
            window: Rolling window for volatility calculation
            annualize: Whether to annualize the volatility
            
        Returns:
            DataFrame with volatility column added
        """
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=window).std()
        
        if annualize:
            df['volatility'] = df['volatility'] * np.sqrt(365)
            
        return df
    
    def get_price_at_date(self, df: pd.DataFrame, target_date: str) -> float:
        """
        Get the closing price at a specific date.
        
        Args:
            df: DataFrame with price data
            target_date: Date in YYYY-MM-DD format
            
        Returns:
            Closing price at the target date
        """
        target = pd.to_datetime(target_date)
        df['date'] = pd.to_datetime(df['date'])
        
        # Find closest date
        idx = (df['date'] - target).abs().idxmin()
        return df.loc[idx, 'close']
    
    def load_cached_data(self, interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Load previously downloaded data from cache.
        
        Args:
            interval: Data interval
            
        Returns:
            DataFrame if cache exists, None otherwise
        """
        filepath = os.path.join(self.data_dir, f"btc_price_history_{interval}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
