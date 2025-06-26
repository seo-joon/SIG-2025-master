import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)
mean_reverting_ids = [2, 5, 11, 18, 37]
# mean_reverting_ids = [8, 25, 33, 37, 48]
#mean_reverting_ids = [2, 4, 6, 20, 47, 19, 22, 29, 1, 16]

def getMyPosition(prcSoFar):
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
