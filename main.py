# main.py
import numpy as np
import pandas as pd
from trading_system import trading_bot
from trading_strategies import strategies
from optimisers import de
from visualisations import visualisation


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)

        if 'close' in df.columns:
            prices = df['close'].values
        else:
            prices = df.iloc[:, 6].values


        prices = prices[::-1]
        print(f" Loaded {len(prices)} points")
        print(f" Start: ${prices[0]:.2f}, End: ${prices[-1]:.2f}")
        print(f" Return: {(prices[-1]/prices[0] - 1)*100:+.1f}%")
        return prices

    except Exception as e:
        print(f"Error: {e}")
        return None


def run_optimization(prices, strategy_name='SMA'):


    print(f"\n{'='*60}")
    print(f"OPTIMIZING {strategy_name} STRATEGY")
    print(f"{'='*60}\n")

    # Choose strategy
    if strategy_name == 'SMA':
        strategy = strategies.SMACrossover()
        config = {
            "pop_size": 30,
            "max_iter": 50,
            "dim": 2,
            "bounds": [(5, 50), (51, 200)],
            "max_time": 120,
            "patience": 15,
            "min_delta": 1.0
        }
    elif strategy_name == 'MACD':
        strategy = strategies.MACDCrossover()
        config = {
            "pop_size": 30,
            "max_iter": 50,
            "dim": 3,
            "bounds": [(5, 30), (20, 60), (5, 20)],  # [fast, slow, signal]
            "max_time": 120,
            "patience": 15,
            "min_delta": 1.0
        }


    bot = trading_bot.TradingBot(prices, strategy)


    optimizer = de.DE(config)

    convergence = []

    print("Starting optimization...")
    best_params = None
    best_fitness = -np.inf

    for iteration,(params, fitness) in enumerate(optimizer.optimise(bot)):
        best_params = params
        best_fitness = fitness
        convergence.append(fitness)

        print(f"Iter {iteration+1:3d}: Fitness = ${fitness:8.2f}, Params = {params}")

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Fitness: ${best_fitness:.2f}")
    print(f"Total Evaluations: {bot.eval_count}")
    print(f"Total Time: {bot.eval_time:.2f}s")
    print(f"Evals/second: {bot.eval_count/bot.eval_time:.1f}")

    return best_params, convergence, bot


if __name__ == "__main__":

    prices = load_data('data/BTC-Hourly.csv')

    if prices is None or len(prices) < 500:
        print("Data loading failed!")




    print(f"\n{'='*60}")
    print("DIAGNOSTIC: Testing Strategy & Backtest")
    print(f"{'='*60}")

    test_strategy = strategies.SMACrossover()
    test_bot = trading_bot.TradingBot(prices, test_strategy)

    test_cases = [
        [10, 50],
        [15, 100],
        [20, 150],
        [30, 180]
    ]

    print("\nTesting parameter combinations:")
    print(f"{'Params':<15} {'Result':<12} {'Status'}")
    print("-" * 50)

    for params in test_cases:
        result = test_bot.evaluate(params)

        if result < 10:
            status = "Critical Failure (ur course)"
        elif result < 500:
            status = "Severe damage (ur WAM)"
        elif result < 900:
            status = " Loss (Hair Loss)"
        elif result > 1100:
            status = " PROFIT [not in ur life]"
        else:
            status = "➖ NEUTRAL"

        print(f"{str(params):<15} ${result:>8.2f}   {status}")


    best_params_sma, convergence_sma, bot_sma = run_optimization(
        prices, strategy_name='SMA'
    )

    visualisation.visualise_results(prices, bot_sma, best_params_sma, convergence_sma, 'SMA')