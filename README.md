# Trading-Bot-Implementation
This project applies nature-inspired optimisation techniques to algorithmic trading. A parameterised strategy using technical indicators (e.g., moving averages) is evaluated on historical Bitcoin OHLCV data, with population-based methods used to optimise parameters through backtesting and comparative analysis

## Project Overview
The goal is to design and analyse intelligent trading bots that can automatically tune strategy parameters and improve performance through adaptive optimisation.

---

## Objectives

- Explore nature-inspired and population-based optimisation techniques  
- Apply optimisation methods to trading strategy parameter tuning  
- Analyse technical indicators for decision-making (buy/sell signals)  
- Evaluate performance using historical Bitcoin market data  
- Compare different optimisation approaches  

---

## Trading Strategy

The trading bot is based on **technical analysis (TA)** indicators such as:

- Simple Moving Average (SMA)  
- Exponential Moving Average (EMA)  
- Indicator crossover signals for buy/sell decisions  

The strategy parameters (e.g., window sizes) are optimised using evolutionary search methods rather than manual tuning.

---

## Optimisation Approach

Nature-inspired optimisation methods are used to:

- Search for optimal trading parameters  
- Improve profitability and reduce risk  
- Navigate a high-dimensional, noisy search space  

These methods operate using populations of candidate solutions that evolve over time toward better performance.

---

## Data

The project uses historical **Bitcoin OHLCV data (Open, High, Low, Close, Volume)**.

This data is used to:

- Simulate trading decisions  
- Backtest strategies  
- Evaluate performance under real market conditions  

---

##  Methodology

1. Define a parameterised trading strategy  
2. Encode parameters as optimisation variables  
3. Use population-based optimisation to search for best parameters  
4. Backtest strategy on historical data  
5. Evaluate performance using trading metrics  

---

##  Phase Breakdown

### Phase 1 – Research
- Study nature-inspired optimisation algorithms  
- Conduct literature review  
- Compare different optimisation approaches  
- Select suitable algorithms for implementation  

### Phase 2 – Implementation
- Implement trading bot  
- Apply optimisation algorithms  
- Perform backtesting  
- Compare results and analyse performance  

---

##  Expected Outcome

- Understanding of nature-inspired optimisation methods  
- A working AI-driven trading bot  
- Comparative analysis of optimisation strategies  
- Performance evaluation on real financial data  

---

## Artificial Bee Colony Optimiser

This section contains the implementation progress for the Artificial Bee Colony algorithm used for Deliverable 2.

### Current Progress

- Implemented the core Artificial Bee Colony optimiser in `optimisers/artificial_bee_colony.py`.
- Added a simple benchmark test in `optimisers/test_abc_optimizer.py` to confirm that the optimiser can improve candidate solutions over iterations.
- Added a convergence visualisation in `visualisations/abc_convergence_plot.py`.
- Generated a convergence plot showing how the best fitness improves across iterations.

### Next Steps

- Connect the ABC optimiser to the Bitcoin trading bot fitness function.
- Run experiments for the 2D, 3D, 7D, 14D and 21D parameter spaces.
- Compare ABC performance against the other algorithms used by the team.

### Notes

The 2D, 3D, 7D, 14D and 21D experiments refer to the number of parameters being optimised in the trading strategy. The project specification describes 7D as a blended signal using weights, durations and EMA alpha; 14D extends this to high-frequency and low-frequency signals; and 21D extends this further for a MACD-style bot. 
