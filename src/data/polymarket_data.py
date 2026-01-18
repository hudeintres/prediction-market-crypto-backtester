"""
Polymarket Binary Options Data Manager

Downloads real Polymarket data or generates synthetic binary options
for backtesting. Note: Polymarket API provides current/active markets
only - historical order book data is not available, so for historical
backtesting we generate synthetic data with realistic mispricing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from scipy.stats import norm
import requests
import os
import json


class PolymarketDataDownloader:
    """
    Downloads real binary options data from Polymarket's API or generates
    synthetic data for backtesting.
    """
    
    GAMMA_API = "https://gamma-api.polymarket.com"
    
    def __init__(self, data_dir: str = "data", use_synthetic: bool = False):
        """
        Initialize the Polymarket data manager.
        
        Args:
            data_dir: Directory to store data
            use_synthetic: Generate synthetic data if API unavailable
        """
        self.data_dir = data_dir
        self.use_synthetic = use_synthetic
        self.polymarket_dir = os.path.join(data_dir, "polymarket")
        os.makedirs(self.polymarket_dir, exist_ok=True)
        self.session = requests.Session()
    
    def get_current_markets(self, limit: int = 100) -> List[Dict]:
        """
        Get currently active markets from Polymarket.
        
        Returns:
            List of market data
        """
        try:
            url = f"{self.GAMMA_API}/markets"
            params = {"limit": limit, "active": "true", "closed": "false"}
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            print(f"Note: Could not fetch Polymarket markets: {e}")
            return []
    
    def get_crypto_price_markets(self) -> pd.DataFrame:
        """
        Get current crypto price prediction markets.
        
        Note: Most historical markets are resolved. For backtesting
        we need to generate synthetic data.
        """
        print("Checking Polymarket for active crypto markets...")
        
        markets = self.get_current_markets(limit=200)
        
        if not markets:
            print("No active markets available from Polymarket API")
            return pd.DataFrame()
        
        crypto_markets = []
        
        for market in markets:
            if not isinstance(market, dict):
                continue
                
            question = str(market.get('question', '')).lower()
            
            # Filter for BTC/ETH price markets
            if any(kw in question for kw in ['bitcoin', 'btc', 'ethereum', 'eth']):
                if any(kw in question for kw in ['price', 'above', 'below', 'reach', '$']):
                    try:
                        # Handle different API response formats
                        yes_price = 0.5
                        if 'outcomePrices' in market and market['outcomePrices']:
                            prices = market['outcomePrices']
                            if isinstance(prices, list) and len(prices) > 0:
                                yes_price = float(prices[0]) if prices[0] else 0.5
                        elif 'yes_bid' in market:
                            yes_price = float(market.get('yes_bid', 0.5))
                        
                        crypto_markets.append({
                            'market_id': market.get('id') or market.get('condition_id', ''),
                            'question': market.get('question', ''),
                            'end_date': market.get('end_date_iso') or market.get('endDate', ''),
                            'volume': float(market.get('volume', 0) or 0),
                            'liquidity': float(market.get('liquidity', 0) or 0),
                            'yes_price': yes_price
                        })
                    except (ValueError, TypeError, KeyError) as e:
                        continue
        
        if crypto_markets:
            df = pd.DataFrame(crypto_markets)
            print(f"Found {len(df)} active crypto price markets on Polymarket")
            return df
        
        print("No active crypto price markets found on Polymarket")
        return pd.DataFrame()
    
    def download_binary_options(
        self,
        spot_prices: pd.DataFrame = None,
        expiry_dates: List[str] = None,
        strikes: List[float] = None,
        base_iv: float = 0.70
    ) -> pd.DataFrame:
        """
        Get binary options data for backtesting.
        
        For current market analysis: fetches from Polymarket API
        For historical backtesting: generates synthetic data with realistic mispricing
        
        Args:
            spot_prices: Historical spot prices (required for synthetic)
            expiry_dates: Expiry dates (required for synthetic)
            strikes: Strike prices (optional)
            base_iv: Base implied volatility
            
        Returns:
            DataFrame with binary options data
        """
        # Check for current market data from Polymarket
        current_markets = self.get_crypto_price_markets()
        
        if not current_markets.empty:
            print(f"Found {len(current_markets)} current markets from Polymarket")
            # Note: Current markets can be used for live trading
            # But for historical backtesting we need synthetic data
        
        # For backtesting historical periods, we need synthetic data
        # since Polymarket doesn't provide historical order books
        if spot_prices is None or expiry_dates is None:
            if self.use_synthetic:
                raise ValueError(
                    "spot_prices and expiry_dates required for synthetic data generation"
                )
            raise RuntimeError(
                "Polymarket API only provides current markets. "
                "For historical backtesting, use --synthetic flag."
            )
        
        print("Generating synthetic binary options for historical backtesting...")
        return self._generate_synthetic_binary_options(
            spot_prices, expiry_dates, strikes, base_iv
        )
    
    def _generate_synthetic_binary_options(
        self,
        spot_prices: pd.DataFrame,
        expiry_dates: List[str],
        strikes: Optional[List[float]] = None,
        base_iv: float = 0.70,
        add_mispricing: bool = True
    ) -> pd.DataFrame:
        """
        Generate synthetic Polymarket-style binary options with realistic mispricing.
        
        Includes behavioral biases observed in prediction markets:
        - Favorite-longshot bias
        - Time-varying noise
        - Occasional large mispricings
        """
        print("[SYNTHETIC] Generating simulated Polymarket binary options...")
        
        if strikes is None:
            spot_range = spot_prices['close'].agg(['min', 'max'])
            min_strike = int(spot_range['min'] * 0.5 / 10000) * 10000
            max_strike = int(spot_range['max'] * 1.5 / 10000) * 10000
            strikes = list(range(max(10000, min_strike), max_strike + 10000, 10000))
        
        binary_options = []
        np.random.seed(42)
        
        for expiry_date in expiry_dates:
            expiry = pd.to_datetime(expiry_date)
            if hasattr(expiry, 'tzinfo') and expiry.tzinfo is not None:
                expiry = expiry.tz_localize(None)
            
            for strike in strikes:
                question = f"Will BTC be above ${strike:,.0f} on {expiry_date}?"
                market_id = f"btc_above_{int(strike)}_{expiry_date.replace('-', '')}"
                
                for idx, row in spot_prices.iterrows():
                    current_date = pd.to_datetime(row['date'])
                    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is not None:
                        current_date = current_date.tz_localize(None)
                    if current_date >= expiry:
                        continue
                    
                    spot = row['close']
                    T = (expiry - current_date).days / 365.0
                    
                    if T <= 0:
                        continue
                    
                    moneyness = np.log(strike / spot)
                    iv = base_iv * (1 + 0.1 * moneyness**2)
                    
                    # Fair price using Black-Scholes (cash-or-nothing call)
                    d1 = (np.log(spot / strike) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
                    d2 = d1 - iv * np.sqrt(T)
                    yes_price = norm.cdf(d2)
                    
                    if add_mispricing:
                        # Favorite-longshot bias
                        if yes_price < 0.3:
                            bias = 0.08 * (0.3 - yes_price)
                        elif yes_price > 0.7:
                            bias = -0.08 * (yes_price - 0.7)
                        else:
                            bias = 0
                        
                        # Time-decay mispricing
                        time_factor = 0.03 + 0.08 * max(0, 1 - T * 4)
                        noise = np.random.normal(0, time_factor)
                        
                        # Occasional large mispricing (creates arbitrage opportunities)
                        if np.random.random() < 0.08:
                            noise += np.random.choice([-0.12, 0.12])
                        
                        yes_price = np.clip(yes_price + bias + noise, 0.02, 0.98)
                    
                    no_price = 1 - yes_price
                    
                    binary_options.append({
                        'date': current_date,
                        'expiry': expiry,
                        'strike': strike,
                        'question': question,
                        'market_id': market_id,
                        'spot': spot,
                        'yes_price': yes_price,
                        'no_price': no_price,
                        'implied_prob': yes_price,
                        'iv': iv,
                        'time_to_expiry': T
                    })
        
        df = pd.DataFrame(binary_options)
        
        filename = os.path.join(self.polymarket_dir, "btc_binary_options.csv")
        df.to_csv(filename, index=False)
        print(f"[SYNTHETIC] Generated {len(df)} binary option snapshots")
        
        return df
    
    def load_binary_options(self) -> Optional[pd.DataFrame]:
        """Load previously generated binary options data."""
        filepath = os.path.join(self.polymarket_dir, "btc_binary_options.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df['expiry'] = pd.to_datetime(df['expiry'])
            return df
        return None


# Alias for backward compatibility
PolymarketDataGenerator = PolymarketDataDownloader
