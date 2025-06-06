import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# The 2 following lines may show errors on import, but the venv is ignoring the errors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

import matplotlib.pyplot as plt

# Loading the dataset
file_path = "../../Dataset/CNN/airPassengers.csv"
df = pd.read_csv(file_path)

# Defining the time index
df["month"] = pd.to_datetime(df["month"])
df.set_index("month", inplace=True)

# Normalizing the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)


# Function to create sequences for time series data training
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    data_len = len(data)

    for i in range(data_len - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


# Defining tge sequence length
sequence_length = 12

# Creating sequences and targets
X, y = create_sequences(df_scaled, sequence_length)

# Splitting the data into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshaping the data for CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Building the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=(sequence_length, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

# Compiling the model
model.compile(optimizer="adam", loss="mse")

# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluating the model
loss = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Data: {loss}")

# Making predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and true values to the original scale
predictions_inv = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test_inv, label="True Values")
plt.plot(df.index[-len(predictions_inv):], predictions_inv, label="Predictions")
plt.title("Air Passengers Time Series Prediction with CNN")
plt.xlabel("Year")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
