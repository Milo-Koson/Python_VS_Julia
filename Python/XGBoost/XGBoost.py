import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../../Dataset/XGBoost/airPassengers.csv"
data = pd.read_csv(file_path)

# Convert the "month" column to datetime format
data["month"] = pd.to_datetime(data["month"])

# Set "month" as the time index
data.set_index("month", inplace=True)

# Plot the time series data from the dataset
plt.figure(figsize=(10, 6))
plt.plot(data, label="Air Passengers")
plt.title("Air Passengers Time Series")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()

# Feature engineering: add time-based features if needed
data["year"] = data.index.year
data["month"] = data.index.month
data["day"] = data.index.day

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]

# Define features and target variable
features = ["year", "month", "day"]
target = "passengers"

# Defining the training and test data
X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# Creating and train the XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# Plot actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, y_test, label="Actual")
plt.plot(test.index, y_pred, label="Predicted")
plt.title("XGBoost Time Series Prediction")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
