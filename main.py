import numpy as np
import pandas as pd
import argparse
from trading_system import trading_bot
from trading_strategies import strategies
from visualisations import visualisation
from optimisers import ga, pso, de


ALGORITHMS = {
    'PSO': pso.PSO,
    'GA':  ga.GA,
    'DE':  de.DE,
}

STRATEGIES = {
    'SMA': {
        'cls':    strategies.SMACrossover,
        'config': {
            "pop_size":  30,
            "max_iter":  50,
            "dim":       2,
            "bounds":    [(5, 50), (51, 200)],
            "max_time":  120,
            "patience":  15,
            "min_delta": 1.0,
        },
        'diagnostics': [[10, 50], [15, 100], [20, 150], [30, 180]],
    },
    'MACD': {
        'cls':    strategies.MACDCrossover,
        'config': {
            "pop_size":  30,
            "max_iter":  50,
            "dim":       3,
            "bounds":    [(5, 30), (20, 60), (5, 20)],
            "max_time":  120,
            "patience":  15,
            "min_delta": 1.0,
        },
        'diagnostics': [[12, 26, 9], [8, 21, 5], [5, 20, 7]],
    },
        'WMA': {                              
        'cls':    strategies.WMACrossover,
        'config': {
            "pop_size":  50,
            "max_iter":  200,
            "dim":       14,
            "bounds":    [
                (0, 1), (0, 1), (0, 1),       # weights HIGH
                (5, 200), (5, 200), (5, 200), # windows HIGH
                (0.01, 0.99),                  # alpha HIGH
                (0, 1), (0, 1), (0, 1),       # weights LOW
                (5, 200), (5, 200), (5, 200), # windows LOW
                (0.01, 0.99),                  # alpha LOW
            ],
            "max_time":  300,
            "patience":  30,
            "min_delta": 1.0,
        },
        'diagnostics': [
            [0.5, 0.3, 0.2, 20, 10, 15, 0.3,
             0.4, 0.4, 0.2, 50, 40, 60, 0.1],
        ],
    },
}



def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        train_mask   = df['date'] < '2020-01-01'
        prices_train = df['close'][train_mask].values
        prices_test  = df['close'][~train_mask].values

        return prices_train, prices_test

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def run_diagnostics(prices_train, selected_strategies):
    print(f"\n{'='*60}")
    print("DIAGNOSTIC: Sanity-checking strategies")
    print(f"{'='*60}")

    for strat_name in selected_strategies:
        entry = STRATEGIES[strat_name]
        print(f"\n--- {strat_name} ---")
        print(f"{'Params':<20} {'Result':<12} {'Status'}")
        print("-" * 55)

        bot = trading_bot.TradingBot(prices_train, entry['cls']())

        for params in entry['diagnostics']:
            result = bot.evaluate(params)
            print(f"{str(params):<20} ${result:>8.2f}   {_classify(result)}")


def _classify(result):
    if result < 10:   return "Critical Failure (ur course)"
    if result < 500:  return "Severe damage (ur WAM)"
    if result < 900:  return "Loss (Hair Loss)"
    if result > 1100: return "PROFIT [not in ur life]"
    return "NEUTRAL"



def run_optimization(prices, strategy_name, algo_name):
    print(f"\n{'='*60}")
    print(f"  {algo_name} x {strategy_name}")
    print(f"{'='*60}\n")

    entry     = STRATEGIES[strategy_name]
    strategy  = entry['cls']()
    config    = entry['config']
    bot       = trading_bot.TradingBot(prices, strategy)
    optimizer = ALGORITHMS[algo_name](config)

    convergence  = []
    best_params  = None
    best_fitness = -np.inf

    for iteration, (params, fitness) in enumerate(optimizer.optimise(bot)):
        best_params  = params
        best_fitness = fitness
        convergence.append(fitness)
        print(f"Iter {iteration+1:3d}: Fitness = ${fitness:8.2f}, Params = {params}")

    print(f"\nBest Fitness:  ${best_fitness:.2f}")
    print(f"Best Params:   {best_params}")
    print(f"Total Evals:   {bot.eval_count}")
    print(f"Total Time:    {bot.eval_time:.2f}s")
    print(f"Evals/second:  {bot.eval_count / bot.eval_time:.1f}")

    return best_params, convergence, bot



def print_summary(results):
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<10} {'Algorithm':<8} {'Train':>12} {'Test':>12}")
    print("-" * 46)

    for (strat, algo), (train_result, test_result) in sorted(results.items()):
        print(f"{strat:<10} {algo:<8} ${train_result:>10.2f} ${test_result:>10.2f}")



def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Trading Bot Optimiser",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --strategies SMA MACD --algorithms PSO GA\n"
            "  python main.py --strategies MACD --algorithms DE\n"
            "  python main.py --no-diagnostics\n"
        )
    )

    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=list(STRATEGIES.keys()),
        default=list(STRATEGIES.keys()),
        metavar='STRATEGY',
        help=f"Strategies to run. Choices: {list(STRATEGIES.keys())} (default: all)"
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        choices=list(ALGORITHMS.keys()),
        default=list(ALGORITHMS.keys()),
        metavar='ALGO',
        help=f"Algorithms to run. Choices: {list(ALGORITHMS.keys())} (default: all)"
    )

    parser.add_argument(
        '--no-diagnostics',
        action='store_true',
        help="Skip the diagnostic sanity-check step"
    )

    parser.add_argument(
        '--data',
        default='data/BTC-Daily.csv',
        help="Path to CSV data file (default: data/BTC-Daily.csv)"
    )

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()

    print(f"\nStrategies : {args.strategies}")
    print(f"Algorithms : {args.algorithms}")

    prices_train, prices_test = load_data(args.data)

    if prices_train is None or len(prices_train) < 500:
        print("Data loading failed!")
        exit()

    if not args.no_diagnostics:
        run_diagnostics(prices_train, args.strategies)

    results = {}

    for strat_name in args.strategies:
        for algo_name in args.algorithms:

            try:
                best_params, convergence, bot_train = run_optimization(
                    prices_train, strat_name, algo_name
                )

                if best_params is None:
                    continue

                bot_test    = trading_bot.TradingBot(prices_test, STRATEGIES[strat_name]['cls']())
                test_result = bot_test.evaluate(best_params)

                print(f"\n[{strat_name} | {algo_name}] Out-of-sample: ${test_result:.2f}")

                results[(strat_name, algo_name)] = (bot_train.evaluate(best_params), test_result)

                visualisation.visualise_results(
                    prices_train, bot_train, best_params, convergence,
                    f"{strat_name}_{algo_name}"
                )

            except Exception as e:
                print(f"\n[{strat_name} | {algo_name}] FAILED: {e}")
                import traceback
                traceback.print_exc()
                continue

    print_summary(results)