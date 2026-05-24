# AI Trading Bot Optimiser

This project implements an AI-driven trading system that uses metaheuristic optimisation algorithms to tune technical trading strategies. It evaluates strategies on historical market data and compares in-sample and out-of-sample performance.

## Overview

The system combines:
- Trading strategies
- Optimisation algorithms
- Backtesting framework
- Train/test split evaluation
- Performance visualisation tools

The goal is to study how different optimisation methods perform across trading strategies and market conditions.

## Features

- Optimisation algorithms:
  - Particle Swarm Optimisation (PSO)
  - Genetic Algorithm (GA)
  - Differential Evolution (DE)
  - Artificial Bee Colony (ABC)

- Trading strategies:
  - 2d_sma – Simple SMA crossover strategy (2D)
  - 3d_macd – Classic MACD crossover strategy (3D)
  - 7d_wma – Weighted moving average crossover (7D)
  - 14d_wma – Extended weighted moving average crossover (14D)
  - 21d_wma – High-dimensional weighted moving average crossover (21D)

- System capabilities:
  - Train/test split (pre-2020 vs post-2020)
  - Diagnostic sanity checks
  - Iterative optimisation tracking
  - Convergence analysis
  - Visualisation of results

## Project Structure

- main.py – entry point
- trading_system.py – trading engine and evaluation
- trading_strategies/ – strategy implementations
- optimisers/ – GA, PSO, DE, ABC algorithms
- visualisations/ – plotting and analysis tools
- data/BTC-Daily.csv – historical dataset

## How to Run

Run full experiment:
```bash
python main.py
```

Run specific strategies and algorithms:
```bash
python main.py --strategies SMA MACD --algorithms PSO GA
```

Skip diagnostics:
```bash
python main.py --no-diagnostics
```

## Output

- Iteration-by-iteration optimisation logs
- Best parameter sets
- Training performance results
- Out-of-sample test performance
- Comparison plots

## Methodology

- Training data: pre-2020
- Testing data: post-2020

Each algorithm optimises strategy parameters to maximise trading returns.

Evaluation metrics:
- Profitability
- Stability
- Convergence behaviour
- Sensitivity to configuration

## Diagnostics

The system runs a diagnostic phase to:
- Validate strategy behaviour
- Test predefined parameter sets
- Detect unstable configurations

## Dependencies

```bash
pip install -r requirements.txt
```

## Contributors
- Dharun Somalingam
- Keerthana Narkunaraja
- Nandana Vinod
- Zi Fung Tan

