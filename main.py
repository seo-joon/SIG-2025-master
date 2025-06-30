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

# potensh
def linearRegression(prcSoFar):
    global currentPos, model_bank, trained
    target = set(range(50))-{20,43,47}
    (nins, nt) = prcSoFar.shape
    newPos = np.zeros(nins)

    if nt <= history_window:
        return newPos

    returns = (prcSoFar[:, 1:nt] - prcSoFar[:, :nt - 1]) / prcSoFar[:, :nt - 1]

    for i in target:
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

        factor = 0.5
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


def rollingZRatio(prcSoFar):
    i, j = 2, 49
    window = 60
    entry_z = 1.5
    exit_z = 0.2
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape

    def zscore(prices, window):
        mean = pd.Series(prices).rolling(window).mean().values
        std = pd.Series(prices).rolling(window).std().values
        return (prices - mean) / (std + 1e-6)

    norm_i = zscore(prcSoFar[i], window)
    norm_j = zscore(prcSoFar[j], window)
    z_ratio = norm_i / (norm_j + 1e-6)

    pos = np.zeros((nInst, nt))
    in_trade = False
    direction = 0

    for t in range(window, nt):
        zi = z_ratio[t]
        pi, pj = prcSoFar[i, t], prcSoFar[j, t]
        mi, mj = int(scale / (pi + 1e-6)), int(scale / (pj + 1e-6))

        if not in_trade:
            if zi > entry_z:
                pos[i, t] = -mi
                pos[j, t] = mj
                in_trade = True
                direction = -1
            elif zi < -entry_z:
                pos[i, t] = mi
                pos[j, t] = -mj
                in_trade = True
                direction = 1
        else:
            if abs(zi) < exit_z:
                in_trade = False
                direction = 0
            else:
                pos[i, t] = direction * -mi
                pos[j, t] = direction * mj

    for k in [i, j]:
        pos[k] = np.clip(pos[k], -int(dlr_limit / (prcSoFar[k, -1] + 1e-6)), int(dlr_limit / (prcSoFar[k, -1] + 1e-6)))

    return pos[:, -1].astype(int)

def get_spread(prices_i, prices_j):
        return np.log(prices_i + 1e-6) - np.log(prices_j + 1e-6)

def zscore(series, window):
    mean = pd.Series(series).rolling(window).mean().values
    std = pd.Series(series).rolling(window).std().values
    return (series - mean) / (std + 1e-6)

def spreadZPairs(prcSoFar):
    window = 60
    entry_z = 1.5
    exit_z = 0.2
    scale = 9000
    dlrPosLimit = 10000
    nInst, nt = prcSoFar.shape
    pos = np.zeros((nInst, nt))
    trade_state = [{"in_trade": False, "direction": 0, "entry_prices": (0, 0), "sizes": (0, 0)} for _ in range(24)]

    for a in range(24):
        i = a
        j = a + 25
        spread = get_spread(prcSoFar[i], prcSoFar[j])
        z = zscore(spread, window)
        state = trade_state[a]

        for t in range(window, nt):
            zval = z[t]
            pi, pj = prcSoFar[i, t], prcSoFar[j, t]
            mi, mj = int(scale / (pi + 1e-6)), int(scale / (pj + 1e-6))

            if not state["in_trade"]:
                if zval > entry_z:
                    pos[i, t] = -mi
                    pos[j, t] = mj
                    state["in_trade"] = True
                    state["direction"] = -1
                    state["entry_prices"] = (pi, pj)
                    state["sizes"] = (-mi, mj)
                elif zval < -entry_z:
                    pos[i, t] = mi
                    pos[j, t] = -mj
                    state["in_trade"] = True
                    state["direction"] = 1
                    state["entry_prices"] = (pi, pj)
                    state["sizes"] = (mi, -mj)
            else:
                pos[i, t] = state["sizes"][0]
                pos[j, t] = state["sizes"][1]
                entry_pi, entry_pj = state["entry_prices"]
                size_i, size_j = state["sizes"]
                pnl = (pi - entry_pi) * size_i + (pj - entry_pj) * size_j
                if abs(zval) < exit_z and pnl > 0:
                    state["in_trade"] = False
                    state["direction"] = 0
                    state["entry_prices"] = (0, 0)
                    state["sizes"] = (0, 0)

        for k in [i, j]:
            pos[k] = np.clip(pos[k], -int(dlrPosLimit / (prcSoFar[k, -1] + 1e-6)), int(dlrPosLimit / (prcSoFar[k, -1] + 1e-6)))

    return pos[:, -1].astype(int)




def zScoreDefault(prcSoFar):
    window = 60
    entry_z = 1.5
    exit_z = 0.2
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    for i in range(24):
        a = i
        b = i + 25
        if nt < window:
            continue

        log_a = np.log(prcSoFar[a, -window:])
        log_b = np.log(prcSoFar[b, -window:])

        X = log_b.reshape(-1, 1)
        y = log_a
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]

        full_spread = np.log(prcSoFar[a] + 1e-6) - beta * np.log(prcSoFar[b] + 1e-6)
        spread = full_spread[-window:]
        mean = np.mean(spread)
        std = np.std(spread) + 1e-6
        z = (full_spread[-1] - mean) / std

        price_a = prcSoFar[a, -1]
        price_b = prcSoFar[b, -1]
        max_a = int(dlr_limit / (price_a + 1e-6))
        max_b = int(dlr_limit / (price_b + 1e-6))

        size_a = int(scale / (price_a + 1e-6))
        size_b = int(scale / (price_b + 1e-6))

        if z > entry_z:
            positions[a] = -min(size_a, max_a)
            positions[b] = min(size_b, max_b)
        elif z < -entry_z:
            positions[a] = min(size_a, max_a)
            positions[b] = -min(size_b, max_b)

    return positions.astype(int)



from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

def betaHedge(prcSoFar):
    window = 60
    entry_z = 1.5
    exit_z = 0.2
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    for i in range(24):
        a = i
        b = i + 25
        if nt < window:
            continue

        log_a = np.log(prcSoFar[a, -window:])
        log_b = np.log(prcSoFar[b, -window:])

        X = log_b.reshape(-1, 1)
        y = log_a
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]

        full_spread = np.log(prcSoFar[a] + 1e-6) - beta * np.log(prcSoFar[b] + 1e-6)
        spread = full_spread[-window:]

        try:
            pval = adfuller(spread)[1]
        except:
            continue

        if pval > 0.05:
            continue

        mean = np.mean(spread)
        std = np.std(spread) + 1e-6
        z = (full_spread[-1] - mean) / std

        price_a = prcSoFar[a, -1]
        price_b = prcSoFar[b, -1]
        max_a = int(dlr_limit / (price_a + 1e-6))
        max_b = int(dlr_limit / (price_b + 1e-6))

        size_a = int(scale / (price_a + 1e-6))
        size_b = int(scale / (price_b + 1e-6))

        if z > entry_z:
            positions[a] = -min(size_a, max_a)
            positions[b] = min(size_b, max_b)
        elif z < -entry_z:
            positions[a] = min(size_a, max_a)
            positions[b] = -min(size_b, max_b)

    return positions.astype(int)

def adfDefault(prcSoFar):
    window = 60
    entry_z = 1.5
    exit_z = 0.2
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    for i in range(24):
        a = i
        b = i + 25
        if nt < window:
            continue

        log_a = np.log(prcSoFar[a, -window:])
        log_b = np.log(prcSoFar[b, -window:])

        X = log_b.reshape(-1, 1)
        y = log_a
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]

        full_spread = np.log(prcSoFar[a] + 1e-6) - beta * np.log(prcSoFar[b] + 1e-6)
        spread = full_spread[-window:]

        spread_lag = spread[:-1]
        spread_ret = np.diff(spread)

        if np.std(spread_lag) == 0:
            continue

        gamma = np.polyfit(spread_lag, spread_ret, 1)[0]
        halflife = -np.log(2) / gamma if gamma < 0 else np.inf

        if halflife < 1 or halflife > 100:
            continue

        mean = np.mean(spread)
        std = np.std(spread) + 1e-6
        z = (full_spread[-1] - mean) / std

        price_a = prcSoFar[a, -1]
        price_b = prcSoFar[b, -1]
        max_a = int(dlr_limit / (price_a + 1e-6))
        max_b = int(dlr_limit / (price_b + 1e-6))

        size_a = int(scale / (price_a + 1e-6))
        size_b = int(scale / (price_b + 1e-6))

        if z > entry_z:
            positions[a] = -min(size_a, max_a)
            positions[b] = min(size_b, max_b)
        elif z < -entry_z:
            positions[a] = min(size_a, max_a)
            positions[b] = -min(size_b, max_b)

    return positions.astype(int)



def shortTermRev(prcSoFar):
    threshold = 0.04
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < 2:
        return positions.astype(int)

    returns = (prcSoFar[:, -1] - prcSoFar[:, -2]) / (prcSoFar[:, -2] + 1e-6)

    for i in range(nInst):
        price = prcSoFar[i, -1]
        size = int(scale / (price + 1e-6))
        max_size = int(dlr_limit / (price + 1e-6))

        if returns[i] > threshold:
            positions[i] = -min(size, max_size)
        elif returns[i] < -threshold:
            positions[i] = min(size, max_size)

    return positions.astype(int)

def recentAvg(prcSoFar):
    window = 10
    threshold = 0.01
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < window + 1:
        return positions.astype(int)

    recent_prices = prcSoFar[:, -window-1:-1]
    mean_price = np.mean(recent_prices, axis=1)
    price = prcSoFar[:, -1]
    gap = (price - mean_price) / (mean_price + 1e-6)

    for i in range(nInst):
        p = price[i]
        size = int(scale / (p + 1e-6))
        max_size = int(dlr_limit / (p + 1e-6))

        if gap[i] > threshold:
            positions[i] = -min(size, max_size)
        elif gap[i] < -threshold:
            positions[i] = min(size, max_size)

    return positions.astype(int)


def RSI(prcSoFar):
    window = 14
    upper_threshold = 70
    lower_threshold = 30
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < window + 1:
        return positions.astype(int)

    delta = np.diff(prcSoFar[:, -window-1:], axis=1)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain, axis=1)
    avg_loss = np.mean(loss, axis=1) + 1e-6
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    price = prcSoFar[:, -1]

    for i in range(nInst):
        p = price[i]
        size = int(scale / (p + 1e-6))
        max_size = int(dlr_limit / (p + 1e-6))

        if rsi[i] > upper_threshold:
            positions[i] = -min(size, max_size)
        elif rsi[i] < lower_threshold:
            positions[i] = min(size, max_size)

    return positions.astype(int)

# good potential
def MACD(prcSoFar):
    fast_period = 12
    slow_period = 26
    signal_period = 9
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < slow_period + signal_period:
        return positions.astype(int)

    price = prcSoFar[:, -1]
    macd = np.zeros(nInst)
    signal = np.zeros(nInst)

    for i in range(nInst):
        fast_ema = pd.Series(prcSoFar[i]).ewm(span=fast_period).mean().values
        slow_ema = pd.Series(prcSoFar[i]).ewm(span=slow_period).mean().values
        macd_line = fast_ema - slow_ema
        signal_line = pd.Series(macd_line).ewm(span=signal_period).mean().values
        macd[i] = macd_line[-1]
        signal[i] = signal_line[-1]

    divergence = macd - signal

    for i in range(nInst):
        p = price[i]
        size = int(scale / (p + 1e-6))
        max_size = int(dlr_limit / (p + 1e-6))

        if divergence[i] > 0:
            positions[i] = min(size, max_size)
        elif divergence[i] < 0:
            positions[i] = -min(size, max_size)

    return positions.astype(int)


# potential
def autoCorr(prcSoFar):
    window = 20
    lag = 1
    threshold = 0.2
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < window + lag:
        return positions.astype(int)

    for i in range(nInst):
        series = prcSoFar[i, -window - lag:]
        x = series[:-lag]
        y = series[lag:]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = np.sum((x - x_mean) * (y - y_mean))
        denom = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2)) + 1e-6
        corr = num / denom

        p = prcSoFar[i, -1]
        size = int(scale / (p + 1e-6))
        max_size = int(dlr_limit / (p + 1e-6))

        if corr > threshold:
            positions[i] = min(size, max_size)
        elif corr < -threshold:
            positions[i] = -min(size, max_size)

    return positions.astype(int)

def autoCorr2(prcSoFar):
    window = 20
    target = set(range(50))-{0,1,2,4,6,16,19,20,25,27,30,33,35,43,44,47}-{3,8,15,18,37,38,40}-{5,7,9,13}-{10,28}#-{32,46}
    #print(f"{set([0, 3, 8, 10, 15, 17, 18, 19, 24, 26, 28, 32, 34, 35, 36, 37, 38, 40, 41, 44, 45, 46, 47]).intersection(set(range(50))-{0,1,2,4,6,16,19,20,25,27,30,33,35,43,44,47}-{3,8,15,18,37,38,40}-{5,7,9,13}-{10,28})}")
    #target = {32, 34, 36, 41, 45, 46, 17, 24, 26}-{32,46}
    #target = {17,26}-{26}
    lag = 1
    threshold = 0.2
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < window + lag:
        return positions.astype(int)

    for i in target:
        series = prcSoFar[i, -window - lag:]
        x = series[:-lag]
        y = series[lag:]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = np.sum((x - x_mean) * (y - y_mean))
        denom = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2)) + 1e-6
        corr = num / denom

        p = prcSoFar[i, -1]
        size = int(scale / (p + 1e-6))
        max_size = int(dlr_limit / (p + 1e-6))

        if corr > threshold:
            positions[i] = min(size, max_size)
        elif corr < -threshold:
            positions[i] = -min(size, max_size)

    return positions.astype(int)



def volSpikes(prcSoFar):
    window = 20
    spike_threshold = 2.0
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < window + 1:
        return positions.astype(int)

    returns = np.diff(prcSoFar[:, -window-1:], axis=1) / (prcSoFar[:, -window-1:-1] + 1e-6)
    vol = np.std(returns[:, :-1], axis=1)
    last_return = returns[:, -1]
    spike = np.abs(last_return) / (vol + 1e-6)

    price = prcSoFar[:, -1]

    for i in range(nInst):
        p = price[i]
        size = int(scale / (p + 1e-6))
        max_size = int(dlr_limit / (p + 1e-6))

        if spike[i] > spike_threshold:
            positions[i] = -min(size, max_size) if last_return[i] > 0 else min(size, max_size)

    return positions.astype(int)


from scipy.stats import skew, kurtosis

def skewTest(prcSoFar):
    window = 30
    skew_thresh = 1.0
    kurt_thresh = 3.5
    scale = 9000
    dlr_limit = 10000
    nInst, nt = prcSoFar.shape
    positions = np.zeros(nInst)

    if nt < window + 1:
        return positions.astype(int)

    returns = np.diff(prcSoFar[:, -window-1:], axis=1) / (prcSoFar[:, -window-1:-1] + 1e-6)

    for i in range(nInst):
        s = skew(returns[i])
        k = kurtosis(returns[i], fisher=False)
        p = prcSoFar[i, -1]
        size = int(scale / (p + 1e-6))
        max_size = int(dlr_limit / (p + 1e-6))

        if s > skew_thresh and k > kurt_thresh:
            positions[i] = -min(size, max_size)
        elif s < -skew_thresh and k > kurt_thresh:
            positions[i] = min(size, max_size)

    return positions.astype(int)


# hello