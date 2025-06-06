# https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Import (download) data
data = pd.read_csv("../../Dataset/Arima/GDPC1.csv")

# Specify GDPC1 series as an ARIMA(2,2,2) model
model = sm.tsa.SARIMAX(data['GDPC1'], order=(2, 2, 2))

# Fit (estimate) the model
results = model.fit()

# Print estimates
print(results.summary())

# Forecast the next 24 values
forecast_values = results.get_forecast(steps=24)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(data['GDPC1'], label='Original Series', linewidth=1)
plt.plot(forecast_values.predicted_mean, label='Prediction', linewidth=1, color='red')
plt.title('ARIMA Prediction')
plt.legend(loc='upper left')
plt.show()
