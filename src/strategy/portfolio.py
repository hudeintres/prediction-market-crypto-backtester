import pandas as pd
"""
Portfolio Management Module

Tracks positions and calculates portfolio-level metrics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class PositionType(Enum):
    """Types of positions in the portfolio."""
    VANILLA_CALL = "vanilla_call"
    VANILLA_PUT = "vanilla_put"
    BINARY_YES = "binary_yes"
    BINARY_NO = "binary_no"


@dataclass
class Position:
    """
    Represents a single position in the portfolio.
    """
    position_id: str
    position_type: PositionType
    
    # Trade details
    entry_date: datetime
    expiry_date: datetime
    strike: float
    quantity: float
    entry_price: float
    
    # State
    is_open: bool = True
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    # PnL
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Associated trade (for linked positions)
    trade_id: Optional[str] = None
    
    @property
    def cost_basis(self) -> float:
        """Total cost to enter position."""
        return self.quantity * self.entry_price
    
    @property
    def current_value(self) -> float:
        """Current value of position."""
        if self.exit_price is not None:
            return self.quantity * self.exit_price
        return self.cost_basis + self.unrealized_pnl
    
    def close(self, exit_date: datetime, exit_price: float) -> float:
        """
        Close the position and calculate realized PnL.
        
        Args:
            exit_date: Date of position close
            exit_price: Price at close
            
        Returns:
            Realized PnL
        """
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.is_open = False
        
        # For vanilla: exit_price is intrinsic value at expiry
        # For binary: exit_price is 0 or 1
        self.realized_pnl = self.quantity * (exit_price - self.entry_price)
        self.unrealized_pnl = 0.0
        
        return self.realized_pnl


@dataclass
class Portfolio:
    """
    Portfolio manager for tracking all positions and calculating metrics.
    """
    name: str = "Vanilla-Binary Arbitrage"
    initial_capital: float = 10000.0
    
    positions: List[Position] = field(default_factory=list)
    cash: float = field(init=False)
    
    # History tracking
    value_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.cash = self.initial_capital
    
    @property
    def open_positions(self) -> List[Position]:
        """Get all open positions."""
        return [p for p in self.positions if p.is_open]
    
    @property
    def closed_positions(self) -> List[Position]:
        """Get all closed positions."""
        return [p for p in self.positions if not p.is_open]
    
    @property
    def total_realized_pnl(self) -> float:
        """Total realized PnL from closed positions."""
        return sum(p.realized_pnl for p in self.closed_positions)
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized PnL from open positions."""
        return sum(p.unrealized_pnl for p in self.open_positions)
    
    @property
    def total_pnl(self) -> float:
        """Total PnL (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl
    
    @property
    def nav(self) -> float:
        """Net Asset Value."""
        open_value = sum(p.current_value for p in self.open_positions)
        return self.cash + open_value
    
    @property
    def return_pct(self) -> float:
        """Total return percentage."""
        return (self.nav - self.initial_capital) / self.initial_capital * 100
    
    def add_position(
        self,
        position_type: PositionType,
        entry_date: datetime,
        expiry_date: datetime,
        strike: float,
        quantity: float,
        entry_price: float,
        trade_id: Optional[str] = None
    ) -> Position:
        """
        Add a new position to the portfolio.
        
        Args:
            position_type: Type of position
            entry_date: Entry date
            expiry_date: Expiry date
            strike: Strike price
            quantity: Quantity
            entry_price: Entry price
            trade_id: Optional trade identifier
            
        Returns:
            Created Position object
        """
        position_id = f"{position_type.value}_{strike}_{pd.to_datetime(expiry_date).strftime('%Y%m%d')}_{len(self.positions)}"
        
        position = Position(
            position_id=position_id,
            position_type=position_type,
            entry_date=entry_date,
            expiry_date=expiry_date,
            strike=strike,
            quantity=quantity,
            entry_price=entry_price,
            trade_id=trade_id
        )
        
        self.positions.append(position)
        self.cash -= position.cost_basis
        
        return position
    
    def close_position(
        self,
        position: Position,
        exit_date: datetime,
        exit_price: float
    ) -> float:
        """
        Close a position.
        
        Args:
            position: Position to close
            exit_date: Exit date
            exit_price: Exit price
            
        Returns:
            Realized PnL
        """
        pnl = position.close(exit_date, exit_price)
        self.cash += position.quantity * exit_price
        return pnl
    
    def close_expiring_positions(
        self,
        date: datetime,
        spot_price: float
    ) -> List[Dict]:
        """
        Close all positions expiring on given date.
        
        Args:
            date: Current/expiry date
            spot_price: Spot price at expiry
            
        Returns:
            List of closed position results
        """
        results = []
        
        for position in self.open_positions:
            if pd.to_datetime(position.expiry_date).date() == date.date():
                # Calculate exit price based on position type
                if position.position_type == PositionType.VANILLA_CALL:
                    exit_price = max(spot_price - position.strike, 0)
                elif position.position_type == PositionType.VANILLA_PUT:
                    exit_price = max(position.strike - spot_price, 0)
                elif position.position_type == PositionType.BINARY_YES:
                    exit_price = 1.0 if spot_price > position.strike else 0.0
                elif position.position_type == PositionType.BINARY_NO:
                    exit_price = 1.0 if spot_price < position.strike else 0.0
                
                pnl = self.close_position(position, date, exit_price)
                
                results.append({
                    'position_id': position.position_id,
                    'position_type': position.position_type.value,
                    'strike': position.strike,
                    'spot_at_expiry': spot_price,
                    'exit_price': exit_price,
                    'pnl': pnl
                })
        
        return results
    
    def update_unrealized_pnl(
        self,
        date: datetime,
        spot_price: float,
        option_pricer: Optional[Any] = None
    ):
        """
        Update unrealized PnL for open positions using proper option pricing.
        
        For hold-to-expiry strategies, we use Black-Scholes estimates rather than
        just intrinsic value. Using only intrinsic value would show misleading
        drawdowns for OTM options that become profitable at expiry.
        
        Args:
            date: Current date
            spot_price: Current spot price
            option_pricer: Optional pricing model for mark-to-market
        """
        from scipy.stats import norm
        
        for position in self.open_positions:
            # Calculate time to expiry
            expiry = pd.to_datetime(position.expiry_date)
            current = pd.to_datetime(date)
            T = max((expiry - current).days / 365.0, 1/365)  # At least 1 day
            
            # Use estimated volatility for MTM (more realistic than intrinsic-only)
            sigma = 0.5  # 50% vol estimate for BTC options
            
            if position.position_type == PositionType.VANILLA_CALL:
                # Black-76 call price
                d1 = (np.log(spot_price / position.strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                current_price = spot_price * norm.cdf(d1) - position.strike * norm.cdf(d2)
                current_price = max(current_price, max(spot_price - position.strike, 0))
                
            elif position.position_type == PositionType.VANILLA_PUT:
                # Black-76 put price
                d1 = (np.log(spot_price / position.strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                current_price = position.strike * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
                current_price = max(current_price, max(position.strike - spot_price, 0))
                
            elif position.position_type == PositionType.BINARY_YES:
                # Binary yes = N(d2) probability estimate
                d1 = (np.log(spot_price / position.strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                current_price = norm.cdf(d2)
                current_price = np.clip(current_price, 0.01, 0.99)
                
            elif position.position_type == PositionType.BINARY_NO:
                # Binary no = 1 - N(d2)
                d1 = (np.log(spot_price / position.strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                current_price = norm.cdf(-d2)
                current_price = np.clip(current_price, 0.01, 0.99)
            
            position.unrealized_pnl = position.quantity * (current_price - position.entry_price)
    
    def record_value(self, date: datetime, spot_price: float):
        """
        Record portfolio value at a point in time.
        
        Args:
            date: Current date
            spot_price: Current spot price
        """
        self.update_unrealized_pnl(date, spot_price)
        
        self.value_history.append({
            'date': date,
            'nav': self.nav,
            'cash': self.cash,
            'open_positions': len(self.open_positions),
            'realized_pnl': self.total_realized_pnl,
            'unrealized_pnl': self.total_unrealized_pnl,
            'spot_price': spot_price
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.value_history:
            return {
                'total_return_pct': self.return_pct,
                'total_pnl': self.total_pnl
            }
        
        df = pd.DataFrame(self.value_history)
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate returns
        df['daily_return'] = df['nav'].pct_change()
        
        # Annualized metrics
        trading_days = len(df)
        annualization_factor = 365 / trading_days if trading_days > 0 else 1
        
        # Use initial capital as base, not first recorded NAV
        # (first NAV may already include unrealized losses from new positions)
        final_nav = df['nav'].iloc[-1]
        total_return = (final_nav - self.initial_capital) / self.initial_capital
        
        # Volatility
        if len(df) > 1:
            daily_vol = df['daily_return'].std()
            annual_vol = daily_vol * np.sqrt(365)
        else:
            annual_vol = 0
        
        # Sharpe ratio (assuming 0 risk-free rate)
        if annual_vol > 0:
            sharpe = (total_return * annualization_factor) / annual_vol
        else:
            sharpe = 0
        
        # Maximum drawdown
        rolling_max = df['nav'].cummax()
        drawdown = (df['nav'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Downside volatility
        negative_returns = df['daily_return'][df['daily_return'] < 0]
        if len(negative_returns) > 0:
            downside_vol = negative_returns.std() * np.sqrt(365)
        else:
            downside_vol = 0
        
        return {
            'total_return_pct': total_return * 100,
            'annualized_return_pct': total_return * annualization_factor * 100,
            'annualized_volatility': annual_vol * 100,
            'downside_volatility': downside_vol * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'total_pnl': self.total_pnl,
            'total_trades': len(self.closed_positions),
            'final_nav': self.nav,
            'trading_days': trading_days
        }
    
    def get_positions_summary(self) -> pd.DataFrame:
        """
        Get summary of all positions.
        
        Returns:
            DataFrame with position details
        """
        data = []
        for p in self.positions:
            data.append({
                'position_id': p.position_id,
                'type': p.position_type.value,
                'strike': p.strike,
                'quantity': p.quantity,
                'entry_price': p.entry_price,
                'entry_date': p.entry_date,
                'expiry_date': p.expiry_date,
                'is_open': p.is_open,
                'exit_price': p.exit_price,
                'realized_pnl': p.realized_pnl,
                'trade_id': p.trade_id
            })
        
        return pd.DataFrame(data)
