# https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.stl.STLForecast.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.datasets import macrodata

ds = macrodata.load_pandas()
data = np.log(ds.data.m1)
base_date = f"{int(ds.data.year[0])}-{3 * int(ds.data.quarter[0]) + 1}-1"
data.index = pd.date_range(base_date, periods=data.shape[0], freq="QS")

# STL-ARIMA forecast
stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 1, 0)})
res = stlf.fit()
forecasts = res.forecast(12)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Series', linewidth=3)
plt.plot(forecasts, label='Forecast', linewidth=3)
plt.title('STL-ARIMA Forecast Prediction (Air Passengers)')
plt.legend(loc='upper left')
plt.show()
