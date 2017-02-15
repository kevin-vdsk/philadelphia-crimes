'''
I create a SARIMAX model for predicting the number of crimes in Philadelphia.
A good reading on SARIMAX models https://tomaugspurger.github.io/modern-7-timeseries.html
For more information about statsmodels:
http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.html
Important here to note is that I have installed released candidate version 0.8.0rc1 of statsmodels.
Let's start importing needed libraries
'''
# Import
import pandas as pd
from collections import namedtuple

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
from statsmodels.tools.eval_measures import rmse

from scipy.optimize import brute

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Group by month and year
# Read CSV
df = pd.read_csv('crime.csv')
print(df.columns)

# Modify month to date time fomat
df['Date'] = pd.to_datetime(df['Month'])

# Derive month and year
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Create time serie of counts
h_ts = df.groupby(['Date']).size()
print(h_ts.head())
h_ts.to_csv('count.csv')

'''
Let's start with a seasonal decomposition.
'''

seas_dec = sm.tsa.seasonal_decompose(h_ts)
seas_dec.plot()
plt.show()

'''
There is clearly a yearly seasonality.
Do we have stationary data ? For answering that, execute the augmented Dickey-Fuller test that basically tells you that:
- H0 (null hypothesis): time serie is non-stationary and it needs to be differenced
- HA (alternative hypothesis): time serie is stationary and it does not need to be differenced
'''

ADF = namedtuple('ADF', 'adf pvalue usedlag nobs critical icbest')    
adf_test = ADF(*adfuller(h_ts))
print(adf_test.critical)
print(adf_test.adf)

'''
Our time serie is not stationary.
'''

def tsplot(y, lags=None, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(-1) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

# Original
tsplot(h_ts, lags=24)
plt.show()

'''
Let's compute now the first difference.
'''
first_diff = h_ts.diff().dropna()
tsplot(first_diff, lags=24)
plt.show()

adf_test = ADF(*adfuller(first_diff))
print(adf_test.critical)
print(adf_test.adf)

'''
It is stationary.
Let's compute now the first difference of seasonal difference.
'''

periods = 12
seasonal_first_diff = (h_ts - h_ts.shift(periods)).diff().dropna()
tsplot(seasonal_first_diff, lags=24)
plt.show()

adf_test = ADF(*adfuller(seasonal_first_diff))
print(adf_test.critical)
print(adf_test.adf)

'''
It is stationary.
Let's fit a model. I want to use only the data up to 2014 for training. Split our data into train and test sets and check it is correctly done.
'''

index_start = h_ts.index.get_loc('2015-01-01')
index_end = h_ts.index.get_loc('2016-01-01')
train_ts = h_ts.iloc[:index_start]
test_ts = h_ts.iloc[index_start:index_end]
unused_ts = h_ts.iloc[index_end:]
print(train_ts.tail())
print(test_ts.head())
print(unused_ts.head)


def objfunc(x, *params):
    model = None
    aic = float('inf')
    rms = float('inf')
    x_int = [int(p.item()) for p in x]
    p, d, q, P, D, Q = x_int
    s, train_ts, test_ts, max_order = params
    if sum(x) <= max_order:
        try:
            # model = smt.SARIMAX(train_ts, trend = 't', order=(p, d, q), seasonal_order=(P, D, Q, s))
            model = smt.SARIMAX(train_ts, order=(p, d, q), seasonal_order=(P, D, Q, s))
            fitted_model = model.fit()
            aic = fitted_model.aic.item()
            forecast = fitted_model.get_forecast(test_ts.size)
            rms = rmse(test_ts, forecast.predicted_mean)
        except:
            print('Error for: ' + str(x))
    return rms

x_grid = (slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2), slice(0, 2))
max_order = 5
s = 12
params = (s, train_ts, test_ts, max_order)
brute_res = brute(objfunc, x_grid, args=params, finish=None, full_output=True)

params = [int(p.item()) for p in brute_res[0]]
rms = brute_res[1]
print('(p, d, q, P, D, Q) = ' + str(params) + ', RMSE = ' + str(rms))

# Re-train with best parameters and display summary
mod_seasonal = smt.SARIMAX(train_ts, order=tuple(params[0:3]), seasonal_order=tuple(params[3:])+(s,))
res_seasonal = mod_seasonal.fit()
print(res_seasonal.summary())
tsplot(res_seasonal.resid, lags=24)
plt.show()

'''
Plot observations and forecast.
'''

pred = res_seasonal.get_prediction(start='2007-01-01', end='2017-12-01')
pred_ci = pred.conf_int()
plt.figure()
ax = train_ts.plot(label='Train', color='green')
ax = test_ts.plot(label='Test', color='red')
ax = unused_ts.plot(label='Unused', color='black')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.legend()
sns.despine()
plt.show()
