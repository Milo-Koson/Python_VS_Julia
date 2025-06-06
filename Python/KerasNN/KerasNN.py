import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# The following 2 lines may show an error, but the venv is ignoring the errors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

# Loading the dataset
file_path = "../../Dataset/Arima/airPassengers.csv"
df = pd.read_csv(file_path)
df["month"] = pd.to_datetime(df["month"])
df = df.set_index("month")

# Processing the Data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Splitting the data into training and testing sets
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]


# Function to create sequences for time series data training
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        seq_in = data[i:i + seq_length]
        seq_out = data[i + seq_length]
        x.append(seq_in)
        y.append(seq_out)
    return np.array(x), np.array(y)


# Define the sequence length
sequence_length = 12

# Creating sequences for training data
x_train, y_train = create_sequences(train_data, sequence_length)

# Creating a simple feedforward neural network model
model = Sequential()
model.add(Dense(50, activation="relu", input_dim=sequence_length))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

# Training the model
model.fit(x_train, y_train, epochs=100, batch_size=1)

# Testing the model on the test data
test_inputs = df_scaled[len(df_scaled) - len(test_data) - sequence_length:]
test_inputs = test_inputs.reshape((1, -1))

x_test = []

for i in range(sequence_length, len(test_inputs[0])):
    x_test.append(test_inputs[0, i - sequence_length:i])

x_test = np.array(x_test)

# Making predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plotting the results
train = df[:train_size]
test = df[train_size:].copy()
test.loc[:, "Predictions"] = predictions

plt.figure(figsize=(12, 6))
plt.plot(train["passengers"], label="Training Data")
plt.plot(test["passengers"], label="Testing Data")
plt.plot(test["Predictions"], label="Predictions")
plt.title("Air Passengers Time Series Prediction")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.legend()
plt.show()
