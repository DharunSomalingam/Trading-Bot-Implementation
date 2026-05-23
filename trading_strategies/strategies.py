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
        fast   = int(params[0])
        slow   = int(params[1])
        signal = int(params[2])

        if fast >= slow:
            return None

        macd_line, signal_line = Indicators.macd(price, fast, slow, signal)

        min_len    = min(len(macd_line), len(signal_line))
        macd_line  = macd_line[-min_len:]
        signal_line = signal_line[-min_len:]

        signals = np.zeros(min_len, dtype=int)

        above = macd_line > signal_line

        for i in range(1, min_len):
            if above[i] and not above[i-1]:
                signals[i] = 1      
            elif not above[i] and above[i-1]:
                signals[i] = -1     

        position = -1
        result   = np.zeros(min_len, dtype=int)

        for i in range(min_len):
            if signals[i] == 1:
                position = 1
            elif signals[i] == -1:
                position = -1
            result[i] = position

        full_signal = np.full(len(price), result[0])
        full_signal[-min_len:] = result

        return full_signal

#7D
class WMACrossover7D(BaseStrategy):
    """
    Optimises only the HIGH (fast) signal as a weighted blend:
        HIGH = (w1·SMA(d1) + w2·LMA(d2) + w3·EMA(d3, alpha)) / (w1+w2+w3)

    LOW (slow) signal is fixed: SMA(200)
    — as per spec page 13, 7-dimensional vector [w1..w3, d1..d3, alpha]

    params = [w1, w2, w3, d1, d2, d3, alpha]
    """
    def generate(self, price, params):
        # ── Unpack 7 params ─────────────────────────────────
        w1, w2, w3 = params[0], params[1], params[2]
        d1 = int(max(5, round(params[3])))
        d2 = int(max(5, round(params[4])))
        d3 = int(max(5, round(params[5])))
        alpha = float(params[6])

        # ── HIGH signal: blended WMA ─────────────────────────
        sma_h = Indicators.sma(price, d1)
        lma_h = Indicators.lma(price, d2)
        ema_h = Indicators.ema(price, d3, alpha)

        min_len = min(len(sma_h), len(lma_h), len(ema_h))
        sma_h = sma_h[-min_len:]
        lma_h = lma_h[-min_len:]
        ema_h = ema_h[-min_len:]

        total_w = w1 + w2 + w3 + 1e-6
        high = (w1*sma_h + w2*lma_h + w3*ema_h) / total_w

        # ── LOW signal: fixed SMA(200) ───────────────────────
        low = Indicators.sma(price, 200)[-min_len:]

        return np.where(high > low, 1, -1)

#14D
class WMACrossover14D(BaseStrategy):
    """    
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

#21D
class WMACrossover21D(BaseStrategy):
    """
    21D blended MACD strategy.
    Three blended WMA lines — FAST, SLOW, SIGNAL — each with 7 parameters.
    
    MACD line   = FAST_WMA − SLOW_WMA
    Signal line = blended WMA of MACD line
    Buy  when MACD line crosses above Signal line
    Sell when MACD line crosses below Signal line

    params = [w1_f, w2_f, w3_f, d1_f, d2_f, d3_f, alpha_f,   ← FAST   (7)
              w1_s, w2_s, w3_s, d1_s, d2_s, d3_s, alpha_s,   ← SLOW   (7)
              w1_g, w2_g, w3_g, d1_g, d2_g, d3_g, alpha_g]   ← SIGNAL (7)
    """
    def generate(self, price, params):
        # ── Unpack FAST (7) ─────────────────────────────────
        w1_f, w2_f, w3_f = params[0], params[1], params[2]
        d1_f = int(max(5, round(params[3])))
        d2_f = int(max(5, round(params[4])))
        d3_f = int(max(5, round(params[5])))
        alpha_f = float(params[6])

        # ── Unpack SLOW (7) ─────────────────────────────────
        w1_s, w2_s, w3_s = params[7], params[8], params[9]
        d1_s = int(max(5, round(params[10])))
        d2_s = int(max(5, round(params[11])))
        d3_s = int(max(5, round(params[12])))
        alpha_s = float(params[13])

        # ── Unpack SIGNAL (7) ───────────────────────────────
        w1_g, w2_g, w3_g = params[14], params[15], params[16]
        d1_g = int(max(5, round(params[17])))
        d2_g = int(max(5, round(params[18])))
        d3_g = int(max(5, round(params[19])))
        alpha_g = float(params[20])

        # ── Build FAST blended WMA ───────────────────────────
        sma_f = Indicators.sma(price, d1_f)
        lma_f = Indicators.lma(price, d2_f)
        ema_f = Indicators.ema(price, d3_f, alpha_f)
        min_f = min(len(sma_f), len(lma_f), len(ema_f))
        sma_f, lma_f, ema_f = sma_f[-min_f:], lma_f[-min_f:], ema_f[-min_f:]
        total_f = w1_f + w2_f + w3_f + 1e-6
        fast_line = (w1_f*sma_f + w2_f*lma_f + w3_f*ema_f) / total_f

        # ── Build SLOW blended WMA ───────────────────────────
        sma_s = Indicators.sma(price, d1_s)
        lma_s = Indicators.lma(price, d2_s)
        ema_s = Indicators.ema(price, d3_s, alpha_s)
        min_s = min(len(sma_s), len(lma_s), len(ema_s))
        sma_s, lma_s, ema_s = sma_s[-min_s:], lma_s[-min_s:], ema_s[-min_s:]
        total_s = w1_s + w2_s + w3_s + 1e-6
        slow_line = (w1_s*sma_s + w2_s*lma_s + w3_s*ema_s) / total_s

        # ── MACD line = FAST − SLOW ──────────────────────────
        min_macd  = min(len(fast_line), len(slow_line))
        fast_line = fast_line[-min_macd:]
        slow_line = slow_line[-min_macd:]
        macd_line = fast_line - slow_line

        # ── Build SIGNAL line (blended WMA of MACD line) ────
        sma_g = Indicators.sma(macd_line, d1_g)
        lma_g = Indicators.lma(macd_line, d2_g)
        ema_g = Indicators.ema(macd_line, d3_g, alpha_g)
        min_g = min(len(sma_g), len(lma_g), len(ema_g))
        sma_g, lma_g, ema_g = sma_g[-min_g:], lma_g[-min_g:], ema_g[-min_g:]
        total_g = w1_g + w2_g + w3_g + 1e-6
        signal_line = (w1_g*sma_g + w2_g*lma_g + w3_g*ema_g) / total_g

        # ── Align MACD and SIGNAL lines ──────────────────────
        min_len     = min(len(macd_line), len(signal_line))
        macd_line   = macd_line[-min_len:]
        signal_line = signal_line[-min_len:]

        # ── Crossover signals ────────────────────────────────
        above   = macd_line > signal_line
        signals = np.zeros(min_len, dtype=int)

        for i in range(1, min_len):
            if above[i] and not above[i-1]:
                signals[i] = 1       # golden cross → buy
            elif not above[i] and above[i-1]:
                signals[i] = -1      # death cross  → sell

        # ── Hold position between crossovers ─────────────────
        position = -1
        result   = np.zeros(min_len, dtype=int)

        for i in range(min_len):
            if signals[i] == 1:
                position = 1
            elif signals[i] == -1:
                position = -1
            result[i] = position

        # ── Pad to match original price length ───────────────
        full_signal = np.full(len(price), result[0])
        full_signal[-min_len:] = result

        return full_signal



