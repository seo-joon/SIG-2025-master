import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import csv
from collections import Counter

nInst = 50
currentPos = np.zeros(nInst)
mean_reverting_ids = [2, 5, 11, 18, 37]
# mean_reverting_ids = [8, 25, 33, 37, 48]
#mean_reverting_ids = [2, 4, 6, 20, 47, 19, 22, 29, 1, 16]

entryThreshold = 0.03
exitThreshold = 0
dlrPosLimit = 10000
history_window = 6
entry_threshold = 0.002
dlrPosLimit = 10000
model_bank = [LinearRegression() for _ in range(nInst)]
trained = [False] * nInst

def read_pairs(csv_path):
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # skip header if any
            if row[0].lower() in ['stock 1', 'stock 2', 'i', 'idx1']: continue
            i, j = int(row[0]), int(row[1])
            pairs.append((i, j))
    return pairs

def pair_mean_reversion_positions(prcSoFar, pairs, lookback=20, scale=5000, zthresh=1.0, invert_signal=False):
    nInst, nt = prcSoFar.shape
    pos = np.zeros(nInst)
    pair_flat = [x for pair in pairs for x in pair]
    pair_count = Counter(pair_flat)
    signal_sign = -1 if not invert_signal else 1

    # aggregate signals
    signals = np.zeros(nInst)
    for (i, j) in pairs:
        if nt < lookback + 1:
            continue
        log_prices_i = np.log(prcSoFar[i, -lookback:])
        log_prices_j = np.log(prcSoFar[j, -lookback:])
        spread = log_prices_i - log_prices_j
        mean_spread = np.mean(spread)
        std_spread = np.std(spread) + 1e-6
        z_score = (spread[-1] - mean_spread) / std_spread
        if np.abs(z_score) > zthresh:
            signals[i] += signal_sign * z_score
            signals[j] -= signal_sign * z_score

    # normalize signals by how many pairs each stock appears in
    for inst in range(nInst):
        if pair_count[inst] > 0:
            signals[inst] /= pair_count[inst]

    # convert signal to position
    for inst in range(nInst):
        pos[inst] = signals[inst] * scale / (prcSoFar[inst, -1] + 1e-6)

    # clip positions to +/- $10,000 per stock
    for k in range(nInst):
        max_shares = int(10000 / (prcSoFar[k, -1] + 1e-6))
        pos[k] = np.clip(pos[k], -max_shares, max_shares)

    return pos.astype(int)

def read_pairs_with_corr(csv_path):
    pairs = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # skip header if needed
            if not row or not row[0].strip().isdigit():
                continue
            i, j = int(row[0]), int(row[1])
            corr = float(row[2])
            pairs.append((i, j, corr))
    return pairs

def pair_mean_reversion_positions_corr(prcSoFar, pairs, lookback=20, scale=5000, zthresh=1.0):
    nInst, nt = prcSoFar.shape
    pos = np.zeros(nInst)
    from collections import Counter
    pair_flat = [x for pair in pairs for x in pair[:2]]
    pair_count = Counter(pair_flat)
    signals = np.zeros(nInst)

    for (i, j, corr) in pairs:
        if nt < lookback + 1:
            continue
        log_prices_i = np.log(prcSoFar[i, -lookback:])
        log_prices_j = np.log(prcSoFar[j, -lookback:])
        spread = log_prices_i - log_prices_j
        mean_spread = np.mean(spread)
        std_spread = np.std(spread) + 1e-6
        z_score = (spread[-1] - mean_spread) / std_spread
        # Flip for negative correlation
        signal_sign = -1 if corr >= 0 else 1
        if np.abs(z_score) > zthresh:
            signals[i] += signal_sign * z_score
            signals[j] -= signal_sign * z_score

    for inst in range(nInst):
        if pair_count[inst] > 0:
            signals[inst] /= pair_count[inst]
        pos[inst] = signals[inst] * scale / (prcSoFar[inst, -1] + 1e-6)
        # Clip positions to $10k per stock
        max_shares = int(10000 / (prcSoFar[inst, -1] + 1e-6))
        pos[inst] = np.clip(pos[inst], -max_shares, max_shares)

    return pos.astype(int)


def getMyPosition(prcSoFar):
    PAIRS = read_pairs_with_corr('high_correlation_pairs.csv')
    return pair_mean_reversion_positions_corr(
        prcSoFar, PAIRS, lookback=65, scale=9000, zthresh=0.5
    )






'''
==================================================
==================================================
'''


def linearRegression(prcSoFar):
    global currentPos, model_bank, trained
    (nins, nt) = prcSoFar.shape
    newPos = np.zeros(nins)

    if nt <= history_window:
        return newPos

    returns = (prcSoFar[:, 1:nt] - prcSoFar[:, :nt - 1]) / prcSoFar[:, :nt - 1]

    for i in range(nins):
        X = np.column_stack([returns[i, j:nt - 1 - history_window + j] for j in range(history_window)])
        y = returns[i, history_window:]

        if X.shape[0] < 1 or y.shape[0] < 1:
            continue

        if not trained[i]:
            model_bank[i].fit(X, y)
            trained[i] = True

        x_pred = returns[i, -history_window:].reshape(1, -1)
        pred = model_bank[i].predict(x_pred)[0]

        price = prcSoFar[i, -1]
        if price == 0:
            continue
        max_pos = int(dlrPosLimit / price)

        factor = 5
        vol_window = 10
        recent_returns = returns[i, -vol_window:]
        if len(recent_returns) < vol_window or np.std(recent_returns) == 0:
            continue

        confidence = pred / np.std(recent_returns)
        raw_size = max_pos * confidence
        size = int(np.clip(raw_size, 0, max_pos))

        factor = 1
        if pred > entry_threshold * factor:
            newPos[i] = size

    currentPos = newPos
    return currentPos

# dailyReturnsMeanReversion
def dailyReturnsMeanReversion(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    newPos = currentPos.copy()

    if nt < 2:
        return newPos

    returns = (prcSoFar[:, -1] - prcSoFar[:, -2]) / prcSoFar[:, -2] # mean reversion property

    for i in range(nins):
        price = prcSoFar[i, -1]
        if price == 0:
            continue
        max_pos = int(dlrPosLimit / price)# * (returns[i]/0.5)) # scaling


        # enter
        if currentPos[i] == 0:
            if returns[i] > entryThreshold:
                newPos[i] = -max_pos
            elif returns[i] < -entryThreshold:
                newPos[i] = max_pos
        else:
            if abs(returns[i]) < exitThreshold:
                newPos[i] = 0

    currentPos = newPos
    return currentPos

def meanReversionBasic(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if nt < 20:
        return np.zeros(nins)

    window = 10
    newPos = np.zeros(nins)

    for i in mean_reverting_ids:
        series = prcSoFar[i, :]
        roll_mean = pd.Series(series).rolling(window=window).mean()
        roll_std = pd.Series(series).rolling(window=window).std()
        z = (pd.Series(series) - roll_mean) / roll_std
        z_val = z.iloc[-2]
        vol_series = pd.Series(series).rolling(window=window).std()
        vol = vol_series.iloc[-2]

        if np.isnan(z_val) or np.isnan(vol):
            continue

        vol_threshold = vol_series.quantile(0.9)
        if abs(z_val) < 1 or vol > vol_threshold:
            signal = 0
        elif z_val > 1.5:
            signal = -1
        elif z_val < -1.5:
            signal = 1
        else:
            signal = 0

        price = series[-1]
        max_position = int(10000 / price)
        newPos[i] = signal * max_position

    currentPos = newPos
    return currentPos
