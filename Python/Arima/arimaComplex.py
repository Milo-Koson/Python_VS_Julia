# https://www.kdnuggets.com/2023/08/times-series-analysis-arima-models-python.html
import warnings
import math
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

net_df = pd.read_csv("../../Dataset/Arima/Netflix_stock_history.csv", index_col="Date", parse_dates=True)

# Show the first 3 rows of the dataset
net_df.head(3)

# Plot the Close, Volume and Open values
net_df[["Close", "Open"]].plot(subplots=True, layout=(1, 2))
plt.show()

train_data, test_data = net_df[0:int(len(net_df) * 0.9)], net_df[int(len(net_df) * 0.9):]

train_arima = train_data['Open']
test_arima = test_data['Open']

history = [x for x in train_arima]
y = test_arima

# Make first prediction
predictions = list()
model = ARIMA(history, order=(1, 1, 0))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])

# Rolling forecasts
for i in range(1, len(y)):
    # predict
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)

# Report performance
mse = mean_squared_error(y, predictions)
print('MSE: ' + str(mse))
mae = mean_absolute_error(y, predictions)
print('MAE: ' + str(mae))
rmse = math.sqrt(mean_squared_error(y, predictions))
print('RMSE: ' + str(rmse))

# Plot the results
plt.figure(figsize=(16, 8))
plt.plot(net_df.index[-600:], net_df['Open'].tail(600), color='green', label='Train Stock Price')
plt.plot(test_data.index, y, color='red', label='Real Stock Price')
plt.plot(test_data.index, predictions, color='blue', label='Predicted Stock Price')
plt.title('Netflix Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Netflix Stock Price')
plt.legend()
plt.grid(True)
plt.show()
