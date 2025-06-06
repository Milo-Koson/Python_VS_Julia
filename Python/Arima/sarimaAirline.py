# https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Getting the data
air_passengers = pd.read_csv("../../Dataset/Arima/airPassengers.csv")

# Converting data for plotting
log_air_passengers = np.log(air_passengers['passengers'])

# Plotting the data
plt.plot(log_air_passengers, label='Original Series', linewidth=1)
plt.title('Original Series')
plt.show()

# Applying SARIMA (Seasonal + ARIMA) to our data
model = sm.tsa.SARIMAX(log_air_passengers, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12))
results = model.fit()

# Print estimates
print(results.summary())

# Getting our forecast on our SARIMA Model
forecast_values = results.get_forecast(steps=24)

# Plotting the forecast results
plt.figure(figsize=(10, 6))
plt.plot(log_air_passengers, label='Original Series', linewidth=1)
plt.plot(forecast_values.predicted_mean, label='Prediction', linewidth=1)
plt.title('SARIMA Prediction')
plt.legend(loc='upper left')
plt.show()
