import numpy as np
from indicators.moving_average import Indicators
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    @abstractmethod
    def generate(self, price, params):
        pass

#2D
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

#3D
class MACDCrossover(BaseStrategy):

    def generate(self, price, params):
        fast = int(params[0])
        slow = int(params[1])
        signal = int(params[2])

        if fast >= slow:
            return None

        macd_line, signal_line = Indicators.macd(price,fast,slow,signal)

        return np.where(macd_line>signal_line,1,-1)

#14D
class WMACrossover(BaseStrategy):
    """
    14D blended WMA crossover strategy.
    
    params = [w1_h, w2_h, w3_h, d1_h, d2_h, d3_h, alpha_h,   ← HIGH signal (7)
              w1_l, w2_l, w3_l, d1_l, d2_l, d3_l, alpha_l]   ← LOW  signal (7)

    HIGH crosses above LOW → buy
    HIGH crosses below LOW → sell
    """
    def generate(self, price, params):
        # ── Unpack ──────────────────────────────────────────
        w1_h, w2_h, w3_h = params[0], params[1], params[2]
        d1_h, d2_h, d3_h = int(max(5, round(params[3]))), \
                            int(max(5, round(params[4]))), \
                            int(max(5, round(params[5])))
        alpha_h           = float(params[6])

        w1_l, w2_l, w3_l = params[7], params[8], params[9]
        d1_l, d2_l, d3_l = int(max(5, round(params[10]))), \
                            int(max(5, round(params[11]))), \
                            int(max(5, round(params[12])))
        alpha_l           = float(params[13])

        # ── Compute component WMAs ───────────────────────────
        sma_h = Indicators.sma(price, d1_h)
        lma_h = Indicators.lma(price, d2_h)
        ema_h = Indicators.ema(price, d3_h, alpha_h)

        sma_l = Indicators.sma(price, d1_l)
        lma_l = Indicators.lma(price, d2_l)
        ema_l = Indicators.ema(price, d3_l, alpha_l)

        # ── Align lengths ────────────────────────────────────
        min_len = min(len(sma_h), len(lma_h), len(ema_h),
                      len(sma_l), len(lma_l), len(ema_l))
        sma_h, lma_h, ema_h = sma_h[-min_len:], lma_h[-min_len:], ema_h[-min_len:]
        sma_l, lma_l, ema_l = sma_l[-min_len:], lma_l[-min_len:], ema_l[-min_len:]

        # ── Blend into HIGH and LOW signals ──────────────────
        total_h = w1_h + w2_h + w3_h + 1e-6
        total_l = w1_l + w2_l + w3_l + 1e-6

        high = (w1_h * sma_h + w2_h * lma_h + w3_h * ema_h) / total_h
        low  = (w1_l * sma_l + w2_l * lma_l + w3_l * ema_l) / total_l

        return np.where(high > low, 1, -1)





