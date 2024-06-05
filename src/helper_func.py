import statsmodels.tsa.stattools as ts
import pandas as pd

def adfuler(data: pd.DataFrame) -> float:
    '''return p value of adfuller test for stationarity of time series data'''
    result = ts.adfuller(data, autolag='AIC')
    return result[1]