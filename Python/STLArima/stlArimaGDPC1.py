import numpy as np
import pandas as pd
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the AirPassengers dataset
air_passengers = pd.read_csv("../../Dataset/Arima/GDPC1.csv")

# Convert the 'month' column to datetime and set it as the index
air_passengers['DATE'] = pd.to_datetime(air_passengers['DATE'])
air_passengers.set_index('DATE', inplace=True)

# Use the 'passengers' column as data
data = np.log(air_passengers['GDPC1'])

# The base date will be the first day of each quarter
base_date = f"{air_passengers.index[0].year}-{3 * (air_passengers.index[0].quarter) + 1}-1"

# Create a date range with quarterly frequency
data.index = pd.date_range(base_date, periods=data.shape[0], freq="QS")

# Initialize the STLForecast model with ARIMA
stlf = STLForecast(data, ARIMA, model_kwargs={"order": (2, 1, 0)})
res = stlf.fit()
forecasts = res.forecast(12)

# Plotting the original data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original Series', linewidth=3)
plt.plot(forecasts, label='Forecast', linewidth=3)
plt.title('STL-ARIMA Forecast Prediction')
plt.legend(loc='upper left')
plt.show()
