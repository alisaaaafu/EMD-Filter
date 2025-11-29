"""
Trading strategies module.

@author: Wanzhen Fu
"""
from .trend_strategy import TrendStrategy, create_trend_order
from .mean_reversion_strategy import MeanReversionStrategy, create_mean_reversion_order

__all__ = [
    'TrendStrategy',
    'create_trend_order',
    'MeanReversionStrategy',
    'create_mean_reversion_order'
]


