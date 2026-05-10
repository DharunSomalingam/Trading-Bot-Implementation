import numpy as np
import time

class TradingBot:
    def __init__(self, price, strategy):
        self.price = price
        self.strategy = strategy
        self.eval_time=0.0
        self.eval_count=0

    def evaluate(self, params):
        signals = self.strategy.generate(self.price, params)

        start_time = time.time()

        if signals is None:
            return -1e9

        signals = np.array(signals)


        final_cash = self._backtest_simple(signals)

        self.eval_time += time.time() - start_time
        self.eval_count += 1

        return final_cash


    def _backtest_simple(self, signals):
        cash = 1000.0
        btc = 0.0
        fee = 0.03


        trade_count = 0

        if len(signals) > 0 and signals[0] == 1:
            btc = (cash * (1 - fee)) / self.price[0]
            cash = 0.0
            trade_count += 1
            current_position = 1
        else:
            current_position = 0

        for idx in range(1, len(self.price)):
            price = self.price[idx]
            prev_signal = signals[idx - 1]
            curr_signal = signals[idx]

            if curr_signal != prev_signal:

                if curr_signal == 1:
                    assert current_position == 0, "Position should be 0 before buy!"
                    assert cash > 0, "Should have cash to buy!"

                    btc = (cash * (1 - fee)) / price
                    cash = 0.0
                    current_position = 1
                    trade_count += 1

                elif curr_signal == -1:
                    # Bearish crossover - SELL
                    assert current_position == 1, "Position should be 1 before sell!"
                    assert btc > 0, "Should have BTC to sell!"

                    cash = btc * price * (1 - fee)
                    btc = 0.0
                    current_position = 0
                    trade_count += 1

        # Final liquidation
        if btc > 0:
            cash = btc * self.price[-1] * (1 - fee)
            trade_count += 1

        return cash



    def backtest_detailed(self, params):
        signals = self.strategy.generate(self.price, params)

        if signals is None:
            return None

        signals = np.array(signals)

        # Alignment
        if len(signals) != len(self.price):
            min_len = min(len(signals), len(self.price))
            signals = signals[-min_len:]
            prices_to_use = self.price[-min_len:]
        else:
            prices_to_use = self.price

        cash = 1000.0
        btc = 0.0
        fee = 0.03

        portfolio_history = []
        trade_log = []

        # Initial position
        if signals[0] == 1:
            btc = (cash * (1 - fee)) / prices_to_use[0]
            trade_log.append({'idx': 0, 'type': 'BUY', 'price': prices_to_use[0]})
            cash = 0.0
            current_position = 1
        else:
            current_position = 0

        portfolio_history.append(cash + btc * prices_to_use[0])

        # Trade on crossovers
        for idx in range(1, len(prices_to_use)):
            price = prices_to_use[idx]
            prev_signal = signals[idx - 1]
            curr_signal = signals[idx]

            if curr_signal != prev_signal:  # Crossover!

                if curr_signal == 1 and current_position == 0:
                    btc = (cash * (1 - fee)) / price
                    trade_log.append({'idx': idx, 'type': 'BUY', 'price': price})
                    cash = 0.0
                    current_position = 1

                elif curr_signal == -1 and current_position == 1:
                    cash = btc * price * (1 - fee)
                    trade_log.append({'idx': idx, 'type': 'SELL', 'price': price})
                    btc = 0.0
                    current_position = 0

            portfolio_value = cash + btc * price
            portfolio_history.append(portfolio_value)

        if btc > 0:
            cash = btc * prices_to_use[-1] * (1 - fee)

        return {
            'signals': signals,
            'portfolio_history': portfolio_history,
            'trade_log': trade_log,
            'final_cash': cash
        }