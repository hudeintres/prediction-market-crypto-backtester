"""
Binary Option Pricing Module

Implements pricing models for cash-or-nothing binary options
as used in prediction markets like Polymarket.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BinaryOptionPrice:
    """Container for binary option pricing results."""
    yes_price: float  # Price of "above" share
    no_price: float   # Price of "below" share
    delta: float      # Sensitivity to spot
    gamma: float      # Sensitivity of delta
    vega: float       # Sensitivity to volatility
    theta: float      # Time decay
    
    @property
    def implied_probability(self) -> float:
        """Implied probability of "yes" outcome."""
        return self.yes_price


class BinaryOptionPricer:
    """
    Pricer for cash-or-nothing binary options.
    
    Binary options pay $1 if a condition is met, $0 otherwise.
    Their fair price represents the risk-neutral probability
    of the condition being met.
    
    For "Will BTC be above $X on date Y?":
    - Yes share = cash-or-nothing call = N(d2)
    - No share = cash-or-nothing put = N(-d2)
    """
    
    def __init__(self, r: float = 0.0):
        """
        Initialize the binary option pricer.
        
        Args:
            r: Risk-free rate (annualized)
        """
        self.r = r
    
    def d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return np.inf if S > K else -np.inf
        return (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return self.d1(S, K, T, sigma) - sigma * np.sqrt(T)
    
    def cash_or_nothing_call(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        payout: float = 1.0
    ) -> float:
        """
        Calculate cash-or-nothing call option price.
        
        Pays fixed amount if S > K at expiry, 0 otherwise.
        This is the fair price for a "Yes" share on a 
        "Will price be above K?" market.
        
        The formula is: e^(-rT) * N(d2) * payout
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            payout: Fixed payout amount (default $1)
            
        Returns:
            Binary call price
        """
        if T <= 0:
            return payout if S > K else 0.0
        
        d2 = self.d2(S, K, T, sigma)
        discount = np.exp(-self.r * T)
        
        return discount * norm.cdf(d2) * payout
    
    def cash_or_nothing_put(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        payout: float = 1.0
    ) -> float:
        """
        Calculate cash-or-nothing put option price.
        
        Pays fixed amount if S < K at expiry, 0 otherwise.
        This is the fair price for a "No" share on a 
        "Will price be above K?" market.
        
        The formula is: e^(-rT) * N(-d2) * payout
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            sigma: Implied volatility
            payout: Fixed payout amount (default $1)
            
        Returns:
            Binary put price
        """
        if T <= 0:
            return payout if S < K else 0.0
        
        d2 = self.d2(S, K, T, sigma)
        discount = np.exp(-self.r * T)
        
        return discount * norm.cdf(-d2) * payout
    
    def implied_probability_above(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate risk-neutral probability of price being above strike at expiry.
        
        This is the theoretical fair price for a binary call with $1 payout.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            
        Returns:
            Probability (0 to 1)
        """
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d2 = self.d2(S, K, T, sigma)
        return norm.cdf(d2)
    
    def implied_probability_below(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate risk-neutral probability of price being below strike at expiry.
        
        This is the theoretical fair price for a binary put with $1 payout.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            
        Returns:
            Probability (0 to 1)
        """
        return 1.0 - self.implied_probability_above(S, K, T, sigma)
    
    def binary_delta(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool = True
    ) -> float:
        """
        Calculate binary option delta.
        
        Delta for binary options behaves differently than vanilla:
        it peaks near the strike and goes to zero far ITM/OTM.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            is_call: True for call, False for put
            
        Returns:
            Binary delta
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d2 = self.d2(S, K, T, sigma)
        discount = np.exp(-self.r * T)
        
        # Binary delta is related to gamma of vanilla
        delta = discount * norm.pdf(d2) / (S * sigma * np.sqrt(T))
        
        return delta if is_call else -delta
    
    def binary_gamma(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> float:
        """
        Calculate binary option gamma.
        
        Binary gamma can be very large near expiry and strike.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            
        Returns:
            Binary gamma
        """
        if T <= 0 or sigma <= 0 or S <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        discount = np.exp(-self.r * T)
        sqrt_T = np.sqrt(T)
        
        # Derivative of delta with respect to S
        term1 = -d1 / (S**2 * sigma * sqrt_T)
        term2 = 1 / (S**2 * sigma**2 * T)
        
        gamma = discount * norm.pdf(d2) * (term1 - term2)
        
        return gamma
    
    def binary_vega(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool = True
    ) -> float:
        """
        Calculate binary option vega.
        
        Unlike vanilla options, binary vega can be negative!
        It depends on moneyness.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            is_call: True for call, False for put
            
        Returns:
            Binary vega (per 1% vol change)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        discount = np.exp(-self.r * T)
        sqrt_T = np.sqrt(T)
        
        # Vega for binary call
        vega = -discount * norm.pdf(d2) * d1 / sigma
        
        if not is_call:
            vega = -vega
        
        return vega / 100  # Per 1% change
    
    def binary_theta(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_call: bool = True
    ) -> float:
        """
        Calculate binary option theta.
        
        Time decay for binary options near the strike can be
        either positive or negative depending on moneyness.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            is_call: True for call, False for put
            
        Returns:
            Binary theta (per day)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        discount = np.exp(-self.r * T)
        sqrt_T = np.sqrt(T)
        
        # Rate of change with time
        term1 = self.r * discount * norm.cdf(d2)
        term2 = discount * norm.pdf(d2) * (
            d1 / (2 * T) - 
            (self.r + 0.5 * sigma**2) / (sigma * sqrt_T)
        )
        
        theta = term1 + term2
        
        if not is_call:
            theta = -theta + self.r * discount
        
        return theta / 365  # Per day
    
    def full_pricing(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float
    ) -> BinaryOptionPrice:
        """
        Calculate full binary option pricing with Greeks.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            
        Returns:
            BinaryOptionPrice with all values
        """
        yes_price = self.cash_or_nothing_call(S, K, T, sigma)
        no_price = self.cash_or_nothing_put(S, K, T, sigma)
        
        return BinaryOptionPrice(
            yes_price=yes_price,
            no_price=no_price,
            delta=self.binary_delta(S, K, T, sigma),
            gamma=self.binary_gamma(S, K, T, sigma),
            vega=self.binary_vega(S, K, T, sigma),
            theta=self.binary_theta(S, K, T, sigma)
        )
    
    def calculate_mispricing(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        sigma: float,
        is_yes_share: bool = True
    ) -> Dict:
        """
        Calculate mispricing between market and fair price.
        
        This is key for identifying arbitrage opportunities.
        
        Args:
            market_price: Observed market price
            S: Spot price
            K: Strike price
            T: Time to expiry
            sigma: Implied volatility
            is_yes_share: True if pricing "yes" (above) share
            
        Returns:
            Dictionary with mispricing analysis
        """
        if is_yes_share:
            fair_price = self.cash_or_nothing_call(S, K, T, sigma)
        else:
            fair_price = self.cash_or_nothing_put(S, K, T, sigma)
        
        absolute = market_price - fair_price
        relative = (market_price - fair_price) / fair_price if fair_price > 0 else 0
        
        return {
            'market_price': market_price,
            'fair_price': fair_price,
            'absolute_mispricing': absolute,
            'relative_mispricing': relative * 100,
            'is_overpriced': absolute > 0.01,
            'is_underpriced': absolute < -0.01,
            'implied_probability_market': market_price,
            'implied_probability_fair': fair_price
        }
    
    def implied_volatility_from_binary(
        self,
        binary_price: float,
        S: float,
        K: float,
        T: float,
        is_call: bool = True,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        """
        Back out implied volatility from binary option price.
        
        Since binary price = N(d2), we can solve for sigma.
        
        Args:
            binary_price: Market price of binary option
            S: Spot price
            K: Strike price
            T: Time to expiry
            is_call: True for call (above), False for put (below)
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
            
        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.5
        
        for _ in range(max_iterations):
            if is_call:
                model_price = self.cash_or_nothing_call(S, K, T, sigma)
            else:
                model_price = self.cash_or_nothing_put(S, K, T, sigma)
            
            diff = model_price - binary_price
            
            if abs(diff) < tolerance:
                return sigma
            
            vega = self.binary_vega(S, K, T, sigma, is_call) * 100
            
            if abs(vega) < 1e-10:
                # Adjust sigma based on diff sign when vega is small
                sigma = sigma * 1.1 if diff < 0 else sigma * 0.9
            else:
                sigma = sigma - diff / vega
            
            sigma = max(0.01, min(5.0, sigma))
        
        return sigma
