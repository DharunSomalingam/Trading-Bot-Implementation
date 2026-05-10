import numpy as np

class Indicators:

    @staticmethod
    def pad(series, n):
        pad = np.full(n - 1, series[0])
        return np.concatenate([pad, series])

    @staticmethod
    def sma(series, n):
        kernel = np.ones(n) / n
        padded = Indicators.pad(series, n)
        return np.convolve(padded, kernel, mode='valid')

    @staticmethod
    def lma(series, n):
        weights = np.array([1 - k/n for k in range(n)])
        kernel = weights * (2 / (n + 1))
        padded = Indicators.pad(series, n)
        return np.convolve(padded, kernel, mode='valid')

    @staticmethod
    def ema(series, n, alpha=None):
        if alpha is None:
            alpha = 2 / (n + 1)

        ema = np.zeros(len(series))
        ema[0] = series[0]

        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]

        return ema

    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        fast_ema = Indicators.ema(series, fast)
        slow_ema = Indicators.ema(series, slow)

        macd_line = fast_ema - slow_ema
        signal_line = Indicators.ema(macd_line, signal)

        return macd_line, signal_line