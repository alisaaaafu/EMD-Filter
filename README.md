# EMD-Filter

An adaptive trading system using Empirical Mode Decomposition (EMD) for quantitative trading in stock index futures markets.

## Overview

This project implements a regime-switching trading strategy that uses EMD to decompose price signals and identify market conditions. The system automatically switches between trend-following and mean-reversion strategies based on the signal-to-noise ratio detected in the market data.

## Features

- **Empirical Mode Decomposition (EMD)**: Custom implementation with cubic spline interpolation
- **Regime Detection**: Automatic identification of trending vs. mean-reverting markets
- **Dual Strategy System**: Trend-following and mean-reversion strategies
- **Backtesting Framework**: Complete backtesting with transaction costs

## Project Structure

```
EMD-Filter/
├── emd/                        # EMD implementation
├── strategies/                 # Trading strategies
├── data/                       # Market data (CSV files)
├── main.py                     # Main trading system
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/alisaaaafu/EMD-Filter.git
cd EMD-Filter
pip install -r requirements.txt
```

**Note**: TA-Lib requires separate installation. See [installation guide](https://github.com/mrjbq7/ta-lib).

## Usage

```bash
python main.py
```

## Data Format

CSV files with OHLC data:
```
Timestamp,Open,High,Low,Close
2018/01/02 09:31,3935.0,3935.4,3930.0,3931.2
```

## Algorithm

1. **EMD Decomposition**: Decompose price signals into Intrinsic Mode Functions (IMFs)
2. **Noise Ratio Calculation**: Compare current vs. historical signal-to-noise ratio
3. **Strategy Selection**:
   - Trend Strategy: Activated when noise is low and trending
   - Mean Reversion Strategy: Activated when noise is high and ranging
4. **Signal Generation**: Use technical indicators for entry/exit

## Results

Output saved to `results/`:
- `market_value.csv`: Portfolio performance
- `continuity_analysis.csv`: Signal quality metrics
- `market_value.png`: Performance chart
- `noise_ratios.png`: EMD ratio visualization

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss.
