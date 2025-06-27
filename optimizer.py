import numpy as np
from pair import pair_mean_reversion_positions_corr
from pair import read_pairs_with_corr
import csv
import itertools
import pandas as pd
def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (nt, nInst, (df.values).T)

def calcPL(prcHist, getPosition, numTestDays=200, commRate=0.0005, dlrPosLimit=10000):
    nInst = prcHist.shape[0]
    nt = prcHist.shape[1]
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:,:t]
        curPrices = prcHistSoFar[:,-1]
        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


def grid_search_with_calcPL_corr(prices, pairs,
                                 lookback_choices=[10, 20, 30, 40],
                                 scale_choices=[2000, 5000, 10000],
                                 zthresh_choices=[0.5, 1.0, 1.5],
                                 numTestDays=200):
    """
    Runs grid search for all parameter combinations, using pair_mean_reversion_positions_corr.
    Assumes pairs contains (i, j, corr).
    """
    param_names = ['lookback', 'scale', 'zthresh']
    param_values = [lookback_choices, scale_choices, zthresh_choices]
    combos = list(itertools.product(*param_values))

    best_score = -np.inf
    best_params = None
    tried = []

    for combo in combos:
        param_dict = dict(zip(param_names, combo))
        def getPosition(prcSoFar, param=param_dict):
            return pair_mean_reversion_positions_corr(
                prcSoFar,
                pairs=pairs,
                lookback=param['lookback'],
                scale=param['scale'],
                zthresh=param['zthresh']
            )
        meanpl, ret, plstd, sharpe, dvol = calcPL(prices, getPosition, numTestDays=numTestDays)
        score = meanpl - 0.1 * plstd
        tried.append((param_dict.copy(), score))
        print(f"Tested {param_dict}, Score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = param_dict.copy()

    print("Grid search completed.")
    print("Best params:", best_params)
    print("Best Score:", best_score)
    return best_params, best_score, tried

def greedy_optimize_with_calcPL_corr(
    prices, pairs,
    lookback_choices=[10, 20, 30, 40],
    scale_choices=[2000, 5000, 10000],
    zthresh_choices=[0.5, 1.0, 1.5],
    numTestDays=200,
    lookback_init=20, scale_init=5000, zthresh_init=1.0,
    max_iters=20
):
    """
    Greedy coordinate ascent optimizer using calcPL and pair_mean_reversion_positions_corr.
    """
    params = {
        'lookback': lookback_init,
        'scale': scale_init,
        'zthresh': zthresh_init
    }
    param_grid = {
        'lookback': lookback_choices,
        'scale': scale_choices,
        'zthresh': zthresh_choices
    }

    def getPosition(prcSoFar, param=params):
        return pair_mean_reversion_positions_corr(
            prcSoFar,
            pairs=pairs,
            lookback=param['lookback'],
            scale=param['scale'],
            zthresh=param['zthresh']
        )

    meanpl, ret, plstd, sharpe, dvol = calcPL(prices, getPosition, numTestDays=numTestDays)
    best_score = meanpl - 0.1 * plstd
    print(f"Initial params: {params}, Score: {best_score:.4f}")
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        for key in params.keys():
            curr_val = params[key]
            for val in param_grid[key]:
                if val == curr_val:
                    continue
                test_params = params.copy()
                test_params[key] = val

                def getPosition(prcSoFar, param=test_params):
                    return pair_mean_reversion_positions_corr(
                        prcSoFar,
                        pairs=pairs,
                        lookback=param['lookback'],
                        scale=param['scale'],
                        zthresh=param['zthresh']
                    )
                meanpl, ret, plstd, sharpe, dvol = calcPL(prices, getPosition, numTestDays=numTestDays)
                score = meanpl - 0.1 * plstd
                print(f"Tested {key}={val}: Score={score:.4f}")
                if score > best_score:
                    print(f"Updated {key}: {curr_val} -> {val} (Score: {best_score:.4f} -> {score:.4f})")
                    params[key] = val
                    best_score = score
                    improved = True
        iters += 1

    print("Greedy optimization completed after", iters, "iterations.")
    print("Best params:", params)
    print("Best Score:", best_score)
    return params, best_score

(nt, nInst, prices) = loadPrices('prices.txt')
pairs = read_pairs_with_corr('high_correlation_pairs.csv')

numTestDays = 750  # Adjust as needed

best_params, best_score = greedy_optimize_with_calcPL_corr(
    prices, pairs,
    lookback_choices=range(10, 150, 1),
    scale_choices=range(2000, 10000, 500),
    zthresh_choices=[0.5, 1.0, 1.5, 2.0, 2.5],
    numTestDays=numTestDays,
    lookback_init=60,
    scale_init=7000,
    zthresh_init=0.5,
    max_iters=20
)

