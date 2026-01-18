"""
Vanilla-Binary Options Arbitrage Strategy - Fixed Version

Implements the arbitrage strategy from:
"Derivative Arbitrage Strategies in Cryptocurrency Markets" by Augustin Valery
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy.stats import norm


@dataclass
class ArbitrageOpportunity:
    """Container for an identified arbitrage opportunity."""
    date: datetime
    expiry: datetime
    vanilla_strike: float
    vanilla_type: str
    vanilla_price_usd: float
    vanilla_price_btc: float
    vanilla_quantity: float
    vanilla_delta: float
    vanilla_iv: float
    binary_strike: float
    binary_type: str
    binary_price: float
    binary_quantity: float
    spot_price: float
    time_to_expiry: float
    total_cost_usd: float
    combined_itm_probability: float
    arbitrage_condition_satisfied: bool
    required_vanilla_strike: float
    max_loss: float
    max_profit: float
    breakeven_price: float


@dataclass  
class TradeResult:
    """Result of a trade at expiration."""
    opportunity: ArbitrageOpportunity
    expiry_price: float
    vanilla_payoff: float
    binary_payoff: float
    total_payoff: float
    gross_return: float
    net_return_pct: float
    vanilla_expired_itm: bool
    binary_expired_itm: bool
    both_expired_itm: bool


class VanillaBinaryArbitrage:
    """
    Implements the vanilla-binary options arbitrage strategy.
    """
    
    def __init__(
        self,
        transaction_fee: float = 0.0003,
        min_combined_probability: float = 0.0,
        capital: float = 10000.0
    ):
        self.transaction_fee = transaction_fee
        self.min_combined_probability = min_combined_probability
        self.capital = capital
        self.opportunities: List[ArbitrageOpportunity] = []
        self.trades: List[TradeResult] = []
    
    def calculate_binary_quantity(
        self,
        vanilla_quantity: float,
        vanilla_price_usd: float,
        binary_price: float,
        include_fees: bool = True
    ) -> float:
        """Calculate required quantity of binary options."""
        fee = self.transaction_fee * vanilla_price_usd * vanilla_quantity if include_fees else 0
        if binary_price >= 1:
            return float('inf')
        qb = vanilla_quantity * (vanilla_price_usd + fee / vanilla_quantity) / (1 - binary_price)
        return qb
    
    def check_arbitrage_condition(
        self,
        vanilla_strike: float,
        binary_strike: float,
        vanilla_price_usd: float,
        binary_price: float,
        vanilla_type: str
    ) -> Tuple[bool, float]:
        """
        Check if arbitrage condition is satisfied.
        
        For Call + Binary Put: KV <= KB - PV/(1-PB)
        For Put + Binary Call: KV >= KB + PV/(1-PB)
        """
        if binary_price >= 0.99 or binary_price <= 0.01:
            return False, 0.0
        
        offset = vanilla_price_usd / (1 - binary_price)
        
        if vanilla_type == 'call':
            # Call + Binary Put: vanilla strike should be below binary strike
            required_strike = binary_strike - offset
            satisfied = vanilla_strike <= required_strike and vanilla_strike < binary_strike
        else:
            # Put + Binary Call: vanilla strike should be above binary strike
            required_strike = binary_strike + offset
            satisfied = vanilla_strike >= required_strike and vanilla_strike > binary_strike
        
        return satisfied, required_strike
    
    def calculate_combined_probability(
        self,
        vanilla_delta: float,
        binary_price: float,
        is_call: bool
    ) -> float:
        """Calculate combined probability of both options expiring ITM."""
        if is_call:
            return abs(vanilla_delta) * (1 - binary_price)
        else:
            return abs(vanilla_delta) * binary_price
    
    def calculate_payoff(
        self,
        spot_at_expiry: float,
        opportunity: ArbitrageOpportunity
    ) -> Dict[str, float]:
        """Calculate payoff at expiration."""
        opp = opportunity
        
        if opp.vanilla_type == 'call':
            vanilla_intrinsic = max(spot_at_expiry - opp.vanilla_strike, 0)
        else:
            vanilla_intrinsic = max(opp.vanilla_strike - spot_at_expiry, 0)
        
        vanilla_payoff = opp.vanilla_quantity * (vanilla_intrinsic - opp.vanilla_price_usd)
        
        if opp.binary_type == 'yes':
            binary_expires_itm = spot_at_expiry > opp.binary_strike
        else:
            binary_expires_itm = spot_at_expiry < opp.binary_strike
        
        binary_settlement = 1.0 if binary_expires_itm else 0.0
        binary_payoff = opp.binary_quantity * (binary_settlement - opp.binary_price)
        
        total_payoff = vanilla_payoff + binary_payoff
        net_return_pct = (total_payoff / opp.total_cost_usd * 100) if opp.total_cost_usd > 0 else 0
        
        return {
            'vanilla_payoff': vanilla_payoff,
            'binary_payoff': binary_payoff,
            'total_payoff': total_payoff,
            'gross_return': total_payoff,
            'net_return_pct': net_return_pct,
            'vanilla_expired_itm': vanilla_intrinsic > 0,
            'binary_expired_itm': binary_expires_itm,
            'both_expired_itm': (vanilla_intrinsic > 0) and binary_expires_itm
        }
    
    def find_opportunities(
        self,
        date: datetime,
        spot_price: float,
        vanilla_options: pd.DataFrame,
        binary_options: pd.DataFrame
    ) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities - FIXED to check BOTH directions!
        """
        opportunities = []
        
        for expiry in binary_options['expiry'].unique():
            binary_at_expiry = binary_options[binary_options['expiry'] == expiry]
            vanilla_at_expiry = vanilla_options[vanilla_options['expiry'] == expiry]
            
            if vanilla_at_expiry.empty:
                continue
            
            for _, binary_row in binary_at_expiry.iterrows():
                binary_strike = binary_row['strike']
                
                # TRY BOTH DIRECTIONS for each binary option
                directions = [
                    ('call', 'no', binary_row['no_price']),   # Call + Binary Put
                    ('put', 'yes', binary_row['yes_price'])   # Put + Binary Call
                ]
                
                for vanilla_type, binary_type, binary_price in directions:
                    # Filter matching vanilla options
                    matching_vanilla = vanilla_at_expiry[
                        vanilla_at_expiry['option_type'] == vanilla_type
                    ]
                    
                    for _, vanilla_row in matching_vanilla.iterrows():
                        vanilla_strike = vanilla_row['strike']
                        vanilla_price_usd = vanilla_row['price_usd']
                        vanilla_delta = vanilla_row['delta']
                        vanilla_iv = vanilla_row['iv']
                        time_to_expiry = vanilla_row['time_to_expiry']
                        
                        if vanilla_price_usd <= 0 or binary_price <= 0.01 or binary_price >= 0.99:
                            continue
                        
                        # Check arbitrage condition
                        condition_met, required_strike = self.check_arbitrage_condition(
                            vanilla_strike=vanilla_strike,
                            binary_strike=binary_strike,
                            vanilla_price_usd=vanilla_price_usd,
                            binary_price=binary_price,
                            vanilla_type=vanilla_type
                        )
                        
                        if not condition_met:
                            continue
                        
                        # Calculate quantities
                        vanilla_quantity = 1.0
                        binary_quantity = self.calculate_binary_quantity(
                            vanilla_quantity=vanilla_quantity,
                            vanilla_price_usd=vanilla_price_usd,
                            binary_price=binary_price
                        )
                        
                        # Calculate combined probability
                        combined_prob = self.calculate_combined_probability(
                            vanilla_delta=vanilla_delta,
                            binary_price=binary_price,
                            is_call=(vanilla_type == 'call')
                        )
                        
                        if combined_prob < self.min_combined_probability:
                            continue
                        
                        total_cost = vanilla_quantity * vanilla_price_usd + binary_quantity * binary_price
                        
                        if vanilla_type == 'call':
                            breakeven = vanilla_strike + vanilla_price_usd + (binary_quantity * binary_price)
                            max_profit = (spot_price * 1.5 - vanilla_strike - vanilla_price_usd) + binary_quantity * (1 - binary_price)
                        else:
                            breakeven = vanilla_strike - vanilla_price_usd - (binary_quantity * binary_price)
                            max_profit = (vanilla_strike - vanilla_price_usd) + binary_quantity * (1 - binary_price)
                        
                        opportunity = ArbitrageOpportunity(
                            date=date,
                            expiry=expiry,
                            vanilla_strike=vanilla_strike,
                            vanilla_type=vanilla_type,
                            vanilla_price_usd=vanilla_price_usd,
                            vanilla_price_btc=vanilla_row['price_btc'],
                            vanilla_quantity=vanilla_quantity,
                            vanilla_delta=vanilla_delta,
                            vanilla_iv=vanilla_iv,
                            binary_strike=binary_strike,
                            binary_type=binary_type,
                            binary_price=binary_price,
                            binary_quantity=binary_quantity,
                            spot_price=spot_price,
                            time_to_expiry=time_to_expiry,
                            total_cost_usd=total_cost,
                            combined_itm_probability=combined_prob,
                            arbitrage_condition_satisfied=True,
                            required_vanilla_strike=required_strike,
                            max_loss=0,
                            max_profit=max_profit,
                            breakeven_price=breakeven
                        )
                        opportunities.append(opportunity)
        
        # Select best opportunity per binary strike (highest combined probability)
        if opportunities:
            df = pd.DataFrame([{
                'idx': i,
                'binary_strike': o.binary_strike,
                'vanilla_type': o.vanilla_type,
                'combined_prob': o.combined_itm_probability
            } for i, o in enumerate(opportunities)])
            
            # Get best per expiry/binary_strike combination
            best_idx = df.groupby('binary_strike')['combined_prob'].idxmax()
            opportunities = [opportunities[i] for i in best_idx.values]
        
        return opportunities
    
    def execute_trade(
        self,
        opportunity: ArbitrageOpportunity,
        spot_at_expiry: float
    ) -> TradeResult:
        """Execute trade and calculate result at expiration."""
        payoff = self.calculate_payoff(spot_at_expiry, opportunity)
        
        result = TradeResult(
            opportunity=opportunity,
            expiry_price=spot_at_expiry,
            vanilla_payoff=payoff['vanilla_payoff'],
            binary_payoff=payoff['binary_payoff'],
            total_payoff=payoff['total_payoff'],
            gross_return=payoff['gross_return'],
            net_return_pct=payoff['net_return_pct'],
            vanilla_expired_itm=payoff['vanilla_expired_itm'],
            binary_expired_itm=payoff['binary_expired_itm'],
            both_expired_itm=payoff['both_expired_itm']
        )
        
        self.trades.append(result)
        return result
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all executed trades."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        returns = [t.net_return_pct for t in self.trades]
        payoffs = [t.total_payoff for t in self.trades]
        
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]
        
        from scipy.stats import skew, kurtosis
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(positive_returns),
            'losing_trades': len(negative_returns),
            'neutral_trades': len(returns) - len(positive_returns) - len(negative_returns),
            'win_rate': len(positive_returns) / len(self.trades) * 100 if self.trades else 0,
            'average_return_pct': np.mean(returns),
            'median_return_pct': np.median(returns),
            'std_return_pct': np.std(returns),
            'max_return_pct': max(returns),
            'min_return_pct': min(returns),
            'total_pnl': sum(payoffs),
            'average_pnl': np.mean(payoffs),
            'both_itm_rate': sum(t.both_expired_itm for t in self.trades) / len(self.trades) * 100,
            'skewness': skew(returns) if len(returns) > 2 else 0,
            'kurtosis': kurtosis(returns) if len(returns) > 3 else 0
        }
