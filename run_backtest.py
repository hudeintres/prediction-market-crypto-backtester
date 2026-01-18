#!/usr/bin/env python3
"""
Prediction Market Crypto Backtester - Main Entry Point

This script downloads BTC price data, options data, and runs
the vanilla-binary arbitrage backtest following the strategy from:
"Derivative Arbitrage Strategies in Cryptocurrency Markets" by Augustin Valery

Usage:
    python run_backtest.py                    # Use real data (default)
    python run_backtest.py --synthetic        # Use synthetic data for testing
    python run_backtest.py --no-cache         # Force fresh data download
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.btc_data import BTCDataDownloader
from src.data.options_data import OptionsDataGenerator
from src.data.polymarket_data import PolymarketDataDownloader
from src.backtester.engine import BacktestEngine, BacktestConfig


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Vanilla-Binary Arbitrage Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_backtest.py                        # Use real data from APIs
  python run_backtest.py --synthetic            # Generate synthetic test data
  python run_backtest.py --no-cache             # Force fresh data download
  python run_backtest.py --start-date 2024-01-01 --end-date 2024-06-01
        """
    )
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='2023-06-01',
        help='Backtest start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-12-31',
        help='Backtest end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital in USD'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory for data storage'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of real API data (for testing)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force fresh data download, ignore cached data'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed trade output'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("PREDICTION MARKET CRYPTO BACKTESTER")
    print("Vanilla-Binary Options Arbitrage Strategy")
    print("="*70)
    
    if args.synthetic:
        print("\n[MODE] Using SYNTHETIC data for testing")
    else:
        print("\n[MODE] Using REAL data from APIs (Yahoo Finance, Deribit, Polymarket)")
    print()
    
    # =========================================================================
    # STEP 1: Download BTC Price History Data
    # =========================================================================
    print("STEP 1: Downloading BTC Price History")
    print("-"*50)
    
    btc_downloader = BTCDataDownloader(
        data_dir=args.data_dir, 
        use_synthetic=args.synthetic
    )
    
    # Try to load cached data first (unless --no-cache)
    spot_prices = None
    if not args.no_cache:
        spot_prices = btc_downloader.load_cached_data()
        if spot_prices is not None:
            print(f"Loaded cached BTC price data: {len(spot_prices)} records")
    
    if spot_prices is None:
        try:
            spot_prices = btc_downloader.download_price_history(
                start_date=args.start_date,
                end_date=args.end_date
            )
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            print("\nTo use synthetic data for testing, run with --synthetic flag:")
            print("  python run_backtest.py --synthetic")
            sys.exit(1)
    
    # Add volatility calculation
    spot_prices = btc_downloader.calculate_volatility(spot_prices, window=30)
    
    print(f"Price range: ${spot_prices['close'].min():,.0f} - ${spot_prices['close'].max():,.0f}")
    print(f"Average 30-day volatility: {spot_prices['volatility'].mean():.1%}")
    print()
    
    # =========================================================================
    # STEP 2: Download/Generate Options Data
    # =========================================================================
    print("STEP 2: Getting BTC Options Data")
    print("-"*50)
    
    options_generator = OptionsDataGenerator(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic
    )
    
    # Try to load cached options data (unless --no-cache)
    vanilla_options = None
    if not args.no_cache:
        vanilla_options = options_generator.load_options_data()
        if vanilla_options is not None:
            print(f"Loaded cached options data: {len(vanilla_options)} records")
    
    if vanilla_options is None:
        # Generate expiry dates
        start = pd.to_datetime(args.start_date)
        end = pd.to_datetime(args.end_date)
        
        all_dates = pd.date_range(start=start, end=end, freq='D')
        expiry_dates = []
        
        for date in all_dates:
            if date.weekday() == 4:  # Friday
                expiry_dates.append(date.strftime('%Y-%m-%d'))
            if date.is_month_end:
                if date.strftime('%Y-%m-%d') not in expiry_dates:
                    expiry_dates.append(date.strftime('%Y-%m-%d'))
        
        expiry_dates = expiry_dates[::4][:24]
        
        print(f"Generating options for {len(expiry_dates)} expiry dates...")
        
        try:
            vanilla_options = options_generator.generate_option_chain(
                spot_prices=spot_prices,
                expiry_dates=expiry_dates,
                strike_range=(0.7, 1.3),
                strike_step=5000,
                base_iv=0.70
            )
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
    
    print(f"Total vanilla options: {len(vanilla_options)}")
    print(f"Unique strikes: {vanilla_options['strike'].nunique()}")
    print(f"Unique expiries: {vanilla_options['expiry'].nunique()}")
    print()
    
    # =========================================================================
    # STEP 3: Download/Generate Polymarket Binary Options Data
    # =========================================================================
    print("STEP 3: Getting Polymarket Binary Options Data")
    print("-"*50)
    
    polymarket_downloader = PolymarketDataDownloader(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic
    )
    
    # Try to load cached binary options data (unless --no-cache)
    binary_options = None
    if not args.no_cache:
        binary_options = polymarket_downloader.load_binary_options()
        if binary_options is not None:
            print(f"Loaded cached binary options data: {len(binary_options)} records")
    
    if binary_options is None:
        unique_expiries = vanilla_options['expiry'].unique()
        expiry_dates = [pd.to_datetime(e).strftime('%Y-%m-%d') for e in unique_expiries]
        
        spot_range = spot_prices['close'].agg(['min', 'max'])
        min_strike = int(spot_range['min'] * 0.7 / 10000) * 10000
        max_strike = int(spot_range['max'] * 1.3 / 10000) * 10000
        binary_strikes = list(range(min_strike, max_strike + 10000, 10000))
        
        print(f"Binary strikes: ${min_strike:,} to ${max_strike:,}")
        
        try:
            binary_options = polymarket_downloader.download_binary_options(
                spot_prices=spot_prices,
                expiry_dates=expiry_dates,
                strikes=binary_strikes,
                base_iv=0.70
            )
        except RuntimeError as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
    
    print(f"Total binary option snapshots: {len(binary_options)}")
    print(f"Unique binary strikes: {binary_options['strike'].nunique()}")
    
    # Print sample markets
    if 'question' in binary_options.columns:
        sample_market = binary_options.groupby(['strike', 'expiry']).first().reset_index()
        print(f"\nSample markets:")
        for _, row in sample_market.head(3).iterrows():
            strike_val = row['strike']
            default_q = 'BTC above $' + f'{strike_val:,.0f}'
            print(f"  - {row.get('question', default_q)}")
    print()
    
    # =========================================================================
    # STEP 4: Run Backtest
    # =========================================================================
    print("STEP 4: Running Backtest")
    print("-"*50)
    
    config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.capital,
        transaction_fee=0.0003,
        min_combined_probability=0.0,
        verbose=not args.quiet
    )
    
    engine = BacktestEngine(config)
    engine.load_data(
        spot_prices=spot_prices,
        vanilla_options=vanilla_options,
        binary_options=binary_options
    )
    
    result = engine.run()
    
    # =========================================================================
    # STEP 5: Print Results
    # =========================================================================
    engine.print_summary(result)
    
    # =========================================================================
    # STEP 6: Export Results
    # =========================================================================
    print("\nSTEP 6: Exporting Results")
    print("-"*50)
    
    results_dir = os.path.join(args.data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not result.daily_values.empty:
        daily_path = os.path.join(results_dir, "daily_portfolio_values.csv")
        result.daily_values.to_csv(daily_path, index=False)
        print(f"Exported daily values to {daily_path}")
    
    if result.trades:
        trades_data = []
        for t in result.trades:
            trades_data.append({
                'date': t.opportunity.date,
                'expiry': t.opportunity.expiry,
                'vanilla_type': t.opportunity.vanilla_type,
                'vanilla_strike': t.opportunity.vanilla_strike,
                'vanilla_price': t.opportunity.vanilla_price_usd,
                'binary_type': t.opportunity.binary_type,
                'binary_strike': t.opportunity.binary_strike,
                'binary_price': t.opportunity.binary_price,
                'expiry_price': t.expiry_price,
                'vanilla_payoff': t.vanilla_payoff,
                'binary_payoff': t.binary_payoff,
                'total_payoff': t.total_payoff,
                'return_pct': t.net_return_pct,
                'vanilla_itm': t.vanilla_expired_itm,
                'binary_itm': t.binary_expired_itm,
                'both_itm': t.both_expired_itm
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_path = os.path.join(results_dir, "trade_details.csv")
        trades_df.to_csv(trades_path, index=False)
        print(f"Exported trade details to {trades_path}")
    
    # Export summary
    summary = {
        'backtest_start': args.start_date,
        'backtest_end': args.end_date,
        'initial_capital': args.capital,
        'data_mode': 'synthetic' if args.synthetic else 'real',
        'total_return_pct': result.total_return_pct,
        'annualized_return_pct': result.annualized_return_pct,
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown_pct': result.max_drawdown_pct,
        'win_rate': result.win_rate,
        'total_trades': len(result.trades),
        'opportunities_found': result.opportunities_found
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(results_dir, "backtest_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Exported summary to {summary_path}")
    
    print("\nBacktest complete!")
    
    return result


if __name__ == "__main__":
    result = main()
