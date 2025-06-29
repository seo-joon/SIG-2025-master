import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

prices = pd.read_csv("prices.txt", sep="\s+", header=None).T
nInst, nDays = prices.shape

def build_features_and_labels(prices, window=10, forecast_horizon=1):
    X = []
    y = []
    inst_idx = []

    for i in range(nInst):
        series = prices.iloc[i].values
        for t in range(window, len(series) - forecast_horizon):
            window_slice = series[t-window:t]
            features = [
                series[t-1],
                np.mean(window_slice),
                np.std(window_slice),
                window_slice[-1] - window_slice[0],
                (window_slice[-1] - np.mean(window_slice)) / (np.std(window_slice) + 1e-6),
            ]
            future_return = (series[t+forecast_horizon] - series[t]) / (series[t] + 1e-6)
            X.append(features)
            y.append(future_return)
            inst_idx.append(i)

    return np.array(X), np.array(y), np.array(inst_idx)

X, y, inst_idx = build_features_and_labels(prices)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R² Score:", round(r2, 6))
print("Mean Squared Error:", round(mse, 6))
