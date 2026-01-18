#!/usr/bin/env python3
"""
Simple Example: Vanilla-Binary Arbitrage Detection

This example demonstrates how to use the backtester to find
arbitrage opportunities between vanilla and binary options.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import norm

from src.pricing.black_scholes import Black76Model
from src.pricing.binary_pricing import BinaryOptionPricer
from src.strategy.arbitrage import VanillaBinaryArbitrage


def main():
    print("="*60)
    print("VANILLA-BINARY ARBITRAGE EXAMPLE")
    print("="*60)
    print()
    
    # Market parameters
    spot = 65000  # BTC spot price
    expiry_days = 30
    T = expiry_days / 365  # Time to expiry in years
    iv = 0.70  # 70% implied volatility
    
    print(f"Market Parameters:")
    print(f"  Spot Price: ${spot:,}")
    print(f"  Time to Expiry: {expiry_days} days")
    print(f"  Implied Volatility: {iv*100}%")
    print()
    
    # Initialize pricers
    black76 = Black76Model(r=0.0)
    binary = BinaryOptionPricer(r=0.0)
    arb = VanillaBinaryArbitrage()
    
    # CALL + NO (binary put) strategy
    binary_strike = 70000  # Above current spot
    
    # Fair prices using Black-Scholes
    d1 = (np.log(spot/binary_strike) + (0 + 0.5*iv**2)*T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    fair_yes = norm.cdf(d2)
    fair_no = 1 - fair_yes
    
    # MISPRICING: Binary NO is underpriced (market inefficiency!)  
    mispriced_no = 0.08  # Underpriced "no" share - arbitrage opportunity!
    mispriced_yes = 1 - mispriced_no
    
    print(f"Binary Option: 'Will BTC be above ${binary_strike:,}?'")
    print(f"  Fair YES Price: {fair_yes:.2f}")
    print(f"  Fair NO Price: {fair_no:.2f}")
    print(f"  Market NO: {mispriced_no:.2f} (UNDERPRICED!)")
    print(f"  Market YES: {mispriced_yes:.2f}")
    print()
    
    # Vanilla call + binary no
    vanilla_strike = 62500  # Below binary strike
    call_price = black76.call_price(spot, vanilla_strike, T, iv)
    
    print(f"Vanilla Call @ ${vanilla_strike:,}")
    print(f"  Price: ${call_price:,.2f}")
    print()
    
    # Check arbitrage condition: KV <= KB - PV/(1-PB)
    condition_met, required_strike = arb.check_arbitrage_condition(
        vanilla_strike=vanilla_strike,
        binary_strike=binary_strike,
        vanilla_price_usd=call_price,
        binary_price=mispriced_no,
        vanilla_type='call'
    )
    
    offset = call_price / (1 - mispriced_no)
    required = binary_strike - offset
    
    print(f"Arbitrage Condition (Call + Binary Put):")
    print(f"  Required KV <= {required:,.0f}")
    print(f"  Actual KV = {vanilla_strike:,}")
    print(f"  Condition Met: {condition_met}")
    print()
    
    if condition_met:
        # Calculate position sizing
        qv = 1  # 1 vanilla option
        qb = qv * call_price / (1 - mispriced_no)
        total_cost = qv * call_price + qb * mispriced_no
        
        print("✓ ARBITRAGE OPPORTUNITY FOUND!")
        print(f"  Buy {qv} vanilla call @ ${vanilla_strike:,}")
        print(f"  Buy {qb:.0f} binary NO contracts @ ${mispriced_no:.2f}")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print()
        
        # Simulate payoffs at different expiry prices
        print("Payoff at Expiration:")
        print("-" * 50)
        for expiry_price in [50000, 60000, 68000, 75000, 90000]:
            vanilla_payoff = qv * (max(expiry_price - vanilla_strike, 0) - call_price)
            # NO wins if price < binary_strike
            binary_wins = expiry_price < binary_strike
            binary_payoff = qb * ((1 if binary_wins else 0) - mispriced_no)
            total_payoff = vanilla_payoff + binary_payoff
            
            marker = "***" if total_payoff > 0 else ""
            print(f"  Spot = ${expiry_price:,}: Total PnL = ${total_payoff:,.2f} {marker}")
        print()
        print("Note: All payoffs are non-negative when arbitrage condition is met!")
        print("This matches the paper's key finding: ZERO losing trades!")
    else:
        print("No arbitrage with current parameters.")
    
    print()
    print("="*60)


if __name__ == "__main__":
    main()
