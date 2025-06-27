#!/usr/bin/env python

import numpy as np
import pandas as pd
from pair import pair_mean_reversion_positions_corr, read_pairs_with_corr

nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

# pricesFile="./priceSlice_test.txt"
pricesFile="./prices.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

def calcPL(prcHist, numTestDays, getPosition):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    hist = []
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
            hist.append((t, value, todayPL, totDVolume, ret, curPos.copy(), posValue, curPrices.copy()))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume, hist)


def getMyPosition(prcSoFar):
    PAIRS = read_pairs_with_corr('high_correlation_pairs.csv')
    return pair_mean_reversion_positions_corr(
        prcSoFar, PAIRS, lookback=135, scale=9500, zthresh=0.5
    )
# Current best params (500 days):
# iter 1: lookback=65, scale=10000, zthresh=0.5 --> score: 27.07
# iter 2: lookback=69, scale=9000, zthresh=0.5 --> score: 29.92
# iter 3: lookback=136, scale=9500, zthresh=0.5 --> score: 47.95

# Current best params (750 days):
# iter 1: lookback=135, scale=9500, zthresh=0.5 --> score: 35.63

(meanpl, ret, plstd, sharpe, dvol, hist) = calcPL(prcAll, getPosition=getMyPosition, numTestDays=750)
score = meanpl - 0.1*plstd
print ("=====")
print ("mean(PL): %.1lf" % meanpl)
print ("return: %.5lf" % ret)
print ("StdDev(PL): %.2lf" % plstd)
print ("annSharpe(PL): %.2lf " % sharpe)
print ("totDvolume: %.0lf " % dvol)
print ("Score: %.2lf" % score)

import matplotlib.pyplot as plt
import seaborn as sns

hist = pd.DataFrame(hist, columns=['Day', 'Value', 'TodayPL', 'TotalDVolume', 'Return', 'Position', 'PositionValue', 'Prices'])

#  Grid of subplots
# subplot 1: Value over time
# subplot 2: Position over time
# subplot 3: returns over time
# subplot 4: All prices over time
# subplot 5: Average returns of all instruments over time
sns.set(style="whitegrid")

fig, axs = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
# Plot Value over time
axs[0].plot(hist['Day'], hist['Value'], label='Portfolio Value', color='blue')
axs[0].set_title('Portfolio Value Over Time')
axs[0].set_ylabel('Value ($)')
axs[0].grid(True)
# Plot Position over time
axs[1].plot(hist['Day'], hist['Position'].apply(np.sum), label='Total Position', color='orange')
axs[1].set_title('Total Position Over Time')
axs[1].set_ylabel('Position Size')
axs[1].grid(True)
# Plot Returns over time
axs[2].plot(hist['Day'], hist['Return'], label='Daily Return', color='green')
axs[2].set_title('Daily Returns Over Time')
axs[2].set_ylabel('Return')
axs[2].set_xlabel('Day')
axs[2].grid(True)
# Plot returns over time
price_hist = {}
for day in hist['Prices']:
    for i, price in enumerate(day):
        if i not in price_hist:
            price_hist[i] = {
                'init_price': price,
                'values': [0]
            }
        else:
            init_value = price_hist[i]['init_price']
            price_hist[i]['values'].append(
                (price - init_value) / init_value
            )
for i in range(nInst):
    axs[3].plot(hist['Day'], price_hist[i]['values'], label=f'Instrument {i}', alpha=0.5)
axs[3].set_title('Cum. Returns of Instruments Over Time')
axs[3].set_ylabel('Returns (%)')
axs[3].set_xlabel('Day')
axs[3].grid(True)
# Plot average returns of all instruments over time
returns_hist = []
stock_init_prices = {}
for i in range(len(hist['Prices'][0])):
    stock_init_prices[i] = hist['Prices'][0][i]
for day in hist['Prices']:
    daily_returns = []
    for i, price in enumerate(day):
        init_price = stock_init_prices[i]
        daily_returns.append((price - init_price) / init_price)
    returns_hist.append(np.mean(daily_returns))
axs[4].plot(hist['Day'], returns_hist, label='Average Returns', color='red')
axs[4].set_title('Average Returns of All Instruments Over Time')
axs[4].set_ylabel('Average Return (%)')
axs[4].set_xlabel('Day')
axs[4].grid(True)


# Adjust layout
plt.tight_layout()
# Show the plots
plt.show()

