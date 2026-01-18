"""
Deribit Options Data Downloader

Downloads real Bitcoin options data from Deribit's public API.
Deribit is the leading crypto options exchange with ~85% market share.

API Documentation: https://docs.deribit.com/

IMPORTANT - DATA AVAILABILITY:
==============================
1. CURRENT DATA (Live): Real-time options with all strikes/expiries (~600+ options)
2. HISTORICAL OPTIONS PRICES: NOT AVAILABLE via public API - only current snapshots
3. HISTORICAL VOLATILITY: ~16 days of hourly HV data available
4. IV SURFACE: Current IV by strike and expiry available
5. SETTLEMENTS: Recent expired option settlements (~12 days back)
6. INDEX PRICES: Historical BTC index prices available (1+ years)

For comprehensive historical options data, consider:
- Tardis.dev (paid, comprehensive crypto derivatives data)
- Kaiko (institutional crypto data)
- Set up scheduled daily snapshots with this tool

The backtester uses real IV/volatility data from Deribit to calibrate the
Black-76 model for generating realistic synthetic historical options prices.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import requests
import time
import os
import json


class DeribitDataDownloader:
    """
    Downloads real BTC options data from Deribit exchange.
    
    Uses Deribit's public REST API (no authentication required for market data).
    """
    
    BASE_URL = "https://www.deribit.com/api/v2/public"
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the Deribit data downloader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.options_dir = os.path.join(data_dir, "deribit_options")
        os.makedirs(self.options_dir, exist_ok=True)
        self.session = requests.Session()
        
    def _api_call(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API call to Deribit."""
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'result' in data:
                return data['result']
            return data
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return {}
    
    def get_index_price(self, currency: str = "BTC") -> Optional[float]:
        """
        Get current BTC index price from Deribit.
        
        Args:
            currency: Currency (BTC or ETH)
            
        Returns:
            Current index price or None
        """
        result = self._api_call("get_index_price", {"index_name": f"{currency.lower()}_usd"})
        return result.get('index_price')
    
    def get_instruments(self, currency: str = "BTC", kind: str = "option", expired: bool = False) -> List[Dict]:
        """
        Get list of available instruments (options).
        
        Args:
            currency: BTC or ETH
            kind: 'option' or 'future'
            expired: Include expired instruments
            
        Returns:
            List of instrument details
        """
        params = {
            "currency": currency,
            "kind": kind,
            "expired": str(expired).lower()
        }
        return self._api_call("get_instruments", params) or []
    
    def get_option_chain(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Get full options chain with current prices.
        
        Args:
            currency: BTC or ETH
            
        Returns:
            DataFrame with all available options
        """
        print(f"Downloading {currency} options chain from Deribit...")
        
        instruments = self.get_instruments(currency, "option", expired=False)
        if not instruments:
            print("No instruments found or API unavailable")
            return pd.DataFrame()
        
        print(f"Found {len(instruments)} active options")
        
        options_data = []
        spot_price = self.get_index_price(currency) or 0
        
        for i, inst in enumerate(instruments):
            # Rate limiting - Deribit allows 10 requests/second for unauthenticated
            if i > 0 and i % 5 == 0:
                time.sleep(0.5)
            
            instrument_name = inst['instrument_name']
            
            # Get order book for this option
            book = self._api_call("get_order_book", {"instrument_name": instrument_name})
            
            if book:
                # Parse instrument name: BTC-DDMMMYY-STRIKE-TYPE
                # Example: BTC-28JUN24-70000-C
                parts = instrument_name.split('-')
                if len(parts) >= 4:
                    try:
                        expiry_str = parts[1]
                        strike = float(parts[2])
                        option_type = 'call' if parts[3] == 'C' else 'put'
                        
                        # Parse expiry date
                        expiry = datetime.strptime(expiry_str, "%d%b%y")
                        
                        options_data.append({
                            'instrument_name': instrument_name,
                            'date': datetime.now(),
                            'expiry': expiry,
                            'strike': strike,
                            'option_type': option_type,
                            'spot': spot_price,
                            'bid_price': book.get('best_bid_price', 0) or 0,
                            'ask_price': book.get('best_ask_price', 0) or 0,
                            'mark_price': book.get('mark_price', 0) or 0,
                            'mark_iv': book.get('mark_iv', 0) or 0,
                            'delta': book.get('greeks', {}).get('delta', 0) or 0,
                            'gamma': book.get('greeks', {}).get('gamma', 0) or 0,
                            'vega': book.get('greeks', {}).get('vega', 0) or 0,
                            'theta': book.get('greeks', {}).get('theta', 0) or 0,
                            'underlying_price': book.get('underlying_price', spot_price),
                            'open_interest': book.get('open_interest', 0) or 0,
                            'volume_24h': inst.get('volume_24h', 0) or 0
                        })
                    except (ValueError, IndexError) as e:
                        continue
        
        df = pd.DataFrame(options_data)
        
        if not df.empty:
            # Calculate time to expiry
            df['time_to_expiry'] = (df['expiry'] - df['date']).dt.days / 365.0
            
            # Price in USD (Deribit quotes in BTC)
            df['price_btc'] = df['mark_price']
            df['price_usd'] = df['mark_price'] * df['spot']
            
            # IV is in percentage, convert to decimal
            df['iv'] = df['mark_iv'] / 100.0
            
            # Save to file
            filename = os.path.join(self.options_dir, f"{currency.lower()}_options_chain.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} options to {filename}")
        
        return df
    
    def get_historical_volatility(self, currency: str = "BTC", period: int = 30) -> Optional[float]:
        """
        Get latest historical volatility from Deribit.
        
        Args:
            currency: BTC or ETH
            period: Period in days (not used - API returns all available)
            
        Returns:
            Latest historical volatility (annualized, as decimal)
        """
        result = self._api_call("get_historical_volatility", {"currency": currency})
        if result:
            # Returns array of [timestamp, hv] pairs
            return result[-1][1] / 100 if result else None
        return None
    
    def get_historical_volatility_series(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Get full historical volatility time series from Deribit (~16 days hourly data).
        
        Args:
            currency: BTC or ETH
            
        Returns:
            DataFrame with timestamp and volatility columns
        """
        print(f"Downloading {currency} historical volatility from Deribit...")
        result = self._api_call("get_historical_volatility", {"currency": currency})
        
        if not result:
            print("No historical volatility data available")
            return pd.DataFrame()
        
        data = []
        for item in result:
            if len(item) >= 2:
                data.append({
                    'timestamp': pd.to_datetime(item[0], unit='ms'),
                    'volatility': item[1] / 100.0  # Convert percentage to decimal
                })
        
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Save to file
            filename = os.path.join(self.options_dir, f"{currency.lower()}_historical_volatility.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} volatility observations ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
            
            # Print summary stats
            print(f"  Average HV: {df['volatility'].mean()*100:.1f}%")
            print(f"  Min/Max: {df['volatility'].min()*100:.1f}% / {df['volatility'].max()*100:.1f}%")
        
        return df
    
    def get_iv_by_expiry(self, currency: str = "BTC") -> Dict[str, float]:
        """
        Get average implied volatility grouped by expiry date.
        
        Returns:
            Dict mapping expiry date string to average IV
        """
        df = self.get_book_summary_by_currency(currency, "option")
        if df.empty:
            return {}
        
        iv_by_expiry = {}
        for expiry in df['expiry'].unique():
            expiry_data = df[df['expiry'] == expiry]
            avg_iv = expiry_data['iv'].mean()
            if not pd.isna(avg_iv) and avg_iv > 0:
                expiry_date = pd.to_datetime(expiry).strftime('%Y-%m-%d')
                iv_by_expiry[expiry_date] = avg_iv
        
        return iv_by_expiry
    
    def get_iv_surface(self, currency: str = "BTC") -> pd.DataFrame:
        """
        Get the current IV surface data (IV by strike and expiry).
        
        Returns:
            DataFrame with strike, expiry, option_type, and iv columns
        """
        df = self.get_book_summary_by_currency(currency, "option")
        if df.empty:
            return pd.DataFrame()
        
        # Create IV surface data
        surface = df[['strike', 'expiry', 'option_type', 'iv', 'spot', 'time_to_expiry']].copy()
        surface['moneyness'] = np.log(surface['strike'] / surface['spot'])
        
        # Save surface data
        filename = os.path.join(self.options_dir, f"{currency.lower()}_iv_surface.csv")
        surface.to_csv(filename, index=False)
        print(f"Saved IV surface with {len(surface)} points")
        
        return surface
    
    def get_calibration_params(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Get market calibration parameters for synthetic data generation.
        
        This extracts key parameters from current Deribit data that can be
        used to make synthetic historical data more realistic.
        
        Returns:
            Dict with calibration parameters:
            - base_iv: Average ATM implied volatility
            - iv_smile_coeff: Volatility smile coefficient (how IV changes with moneyness)
            - term_structure: IV term structure (how IV changes with time to expiry)
            - hv: Historical volatility
        """
        print("Extracting market calibration parameters from Deribit...")
        
        params = {
            'base_iv': 0.70,  # Default fallback
            'iv_smile_coeff': 0.1,
            'term_structure_slope': 0.0,
            'hv': 0.50
        }
        
        # Get historical volatility
        hv = self.get_historical_volatility(currency)
        if hv:
            params['hv'] = hv
            print(f"  Historical volatility: {hv*100:.1f}%")
        
        # Get options data for IV surface
        df = self.get_book_summary_by_currency(currency, "option")
        if df.empty:
            print("  Using default parameters (no live options data)")
            return params
        
        spot = df['spot'].iloc[0] if 'spot' in df.columns else None
        if spot:
            print(f"  Current spot: ${spot:,.0f}")
        
        # Calculate base IV (ATM options)
        if 'strike' in df.columns and 'iv' in df.columns and spot:
            # ATM is where strike ~ spot
            atm_range = (df['strike'] > spot * 0.95) & (df['strike'] < spot * 1.05)
            atm_options = df[atm_range]
            if not atm_options.empty:
                params['base_iv'] = atm_options['iv'].mean()
                print(f"  Base IV (ATM): {params['base_iv']*100:.1f}%")
        
        # Calculate IV smile coefficient (simplified)
        if 'iv' in df.columns and 'strike' in df.columns and spot:
            df['moneyness'] = np.log(df['strike'] / spot)
            otm_call = df[(df['option_type'] == 'call') & (df['moneyness'] > 0.1)]
            otm_put = df[(df['option_type'] == 'put') & (df['moneyness'] < -0.1)]
            
            avg_otm_iv = pd.concat([otm_call['iv'], otm_put['iv']]).mean()
            atm_iv = params['base_iv']
            
            if not pd.isna(avg_otm_iv) and atm_iv > 0:
                # Smile coefficient roughly = (OTM_IV - ATM_IV) / average_moneyness^2
                params['iv_smile_coeff'] = (avg_otm_iv - atm_iv) / (0.15**2)
                print(f"  IV smile coefficient: {params['iv_smile_coeff']:.3f}")
        
        # Calculate term structure
        if 'time_to_expiry' in df.columns and 'iv' in df.columns:
            short_term = df[df['time_to_expiry'] < 30/365]['iv'].mean()
            long_term = df[df['time_to_expiry'] > 90/365]['iv'].mean()
            
            if not pd.isna(short_term) and not pd.isna(long_term):
                # Negative slope = backwardation (short term IV > long term)
                params['term_structure_slope'] = (long_term - short_term)
                print(f"  Term structure slope: {params['term_structure_slope']*100:+.1f}%")
        
        return params
    
    def get_book_summary_by_currency(self, currency: str = "BTC", kind: str = "option") -> pd.DataFrame:
        """
        Get summary of all option books for a currency.
        
        This is more efficient than fetching individual order books.
        
        Args:
            currency: BTC or ETH
            kind: 'option' or 'future'
            
        Returns:
            DataFrame with book summaries
        """
        print(f"Downloading {currency} options summary from Deribit...")
        
        result = self._api_call("get_book_summary_by_currency", {
            "currency": currency,
            "kind": kind
        })
        
        if not result:
            print("No data returned from Deribit API")
            return pd.DataFrame()
        
        options_data = []
        spot_price = self.get_index_price(currency) or 0
        
        for book in result:
            instrument_name = book.get('instrument_name', '')
            parts = instrument_name.split('-')
            
            if len(parts) >= 4 and kind == "option":
                try:
                    expiry_str = parts[1]
                    strike = float(parts[2])
                    option_type = 'call' if parts[3] == 'C' else 'put'
                    expiry = datetime.strptime(expiry_str, "%d%b%y")
                    
                    mark_price = book.get('mark_price', 0) or 0
                    
                    options_data.append({
                        'instrument_name': instrument_name,
                        'date': datetime.now(),
                        'expiry': expiry,
                        'strike': strike,
                        'option_type': option_type,
                        'spot': spot_price,
                        'bid_price': book.get('bid_price', 0) or 0,
                        'ask_price': book.get('ask_price', 0) or 0,
                        'mark_price': mark_price,
                        'mark_iv': book.get('mark_iv', 0) or 0,
                        'price_btc': mark_price,
                        'price_usd': mark_price * spot_price,
                        'open_interest': book.get('open_interest', 0) or 0,
                        'volume_24h': book.get('volume_24h', 0) or 0
                    })
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(options_data)
        
        if not df.empty:
            df['time_to_expiry'] = (df['expiry'] - df['date']).dt.days / 365.0
            df['iv'] = df['mark_iv'] / 100.0
            
            # Estimate delta using Black-76 if not provided
            from scipy.stats import norm
            for idx, row in df.iterrows():
                if row['time_to_expiry'] > 0 and row['iv'] > 0:
                    d1 = (np.log(row['spot'] / row['strike']) + 0.5 * row['iv']**2 * row['time_to_expiry']) / (row['iv'] * np.sqrt(row['time_to_expiry']))
                    if row['option_type'] == 'call':
                        df.loc[idx, 'delta'] = norm.cdf(d1)
                    else:
                        df.loc[idx, 'delta'] = -norm.cdf(-d1)
            
            filename = os.path.join(self.options_dir, f"{currency.lower()}_options_summary.csv")
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} options to {filename}")
        
        return df
    
    def load_cached_options(self, currency: str = "BTC") -> Optional[pd.DataFrame]:
        """
        Load previously downloaded options data.
        
        Args:
            currency: BTC or ETH
            
        Returns:
            DataFrame if cache exists, None otherwise
        """
        for filename in [f"{currency.lower()}_options_chain.csv", f"{currency.lower()}_options_summary.csv"]:
            filepath = os.path.join(self.options_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df['expiry'] = pd.to_datetime(df['expiry'])
                return df
        return None
    
    def download_options_for_backtest(
        self,
        currency: str = "BTC",
        use_summary: bool = True
    ) -> pd.DataFrame:
        """
        Download options data formatted for backtesting.
        
        Args:
            currency: BTC or ETH
            use_summary: Use summary endpoint (faster) vs full chain
            
        Returns:
            DataFrame formatted for backtester
        """
        if use_summary:
            df = self.get_book_summary_by_currency(currency, "option")
        else:
            df = self.get_option_chain(currency)
        
        if df.empty:
            raise RuntimeError(
                "Failed to download options data from Deribit. "
                "Please check your internet connection or use --synthetic flag."
            )
        
        # Format for backtester
        df = df.rename(columns={
            'mark_iv': 'iv_pct'
        })
        
        return df
