import numpy as np
from indicators.moving_average import Indicators
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def generate(self, price, params):
        pass


class SMACrossover(BaseStrategy):

    def generate(self, price, params):
        short_w = int(params[0])
        long_w = int(params[1])

        if short_w >= long_w:
            return None

        short_ma = Indicators.sma(price,short_w)
        long_ma  = Indicators.sma(price,long_w)

        min_len  = min(len(short_ma),len(long_ma))
        short_ma = short_ma[-min_len:]
        long_ma  = long_ma[-min_len:]

        return np.where(short_ma>long_ma,1,-1)



class MACDCrossover(BaseStrategy):

    def generate(self, price, params):
        fast = int(params[0])
        slow = int(params[1])
        signal = int(params[2])

        if fast >= slow:
            return None

        macd_line, signal_line = Indicators.macd(price,fast,slow,signal)

        return np.where(macd_line>signal_line,1,-1)








