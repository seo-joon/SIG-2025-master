import numpy as np
import csv
from collections import Counter

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
    
    
