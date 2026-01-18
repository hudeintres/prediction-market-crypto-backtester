"""
Data module for downloading and processing market data.
"""

from .btc_data import BTCDataDownloader
from .options_data import OptionsDataGenerator
from .polymarket_data import PolymarketDataDownloader, PolymarketDataGenerator

# Optional: Deribit integration
try:
    from .deribit_data import DeribitDataDownloader
except ImportError:
    DeribitDataDownloader = None
