using Flux, Random, CSV, DataFrames, StatsBase, Dates

# Load the data
data = CSV.read("Dataset/LSTM/airPassengers.csv", DataFrame)

# Convert the month column to a Date column
data[!, :month] = Date.(DateTime.(string.(data[!, :month]), "Y-m-d"), "Y-m-d")

# Define the window size
window_size = 12

# Determine the number of samples in each set
train_size = Int(floor(0.7 * size(data, 1)))
test_size = size(data, 1) - train_size

# Create input and output matrices for the training set
X_train = zeros(train_size, window_size)
y_train = zeros(train_size)

# Create input and output matrices for the test set
X_test = zeros(test_size, window_size)
y_test = zeros(test_size)

# Populate the matrices X_train, y_train, X_test and y_test with data
for i in 1:train_size
   X_train[i, :] = [Dates.year(data[i+j-1, :month]) * 100 + Dates.month(data[i+j-1, :month]) for j in 1:window_size]
   y_train[i] = Dates.year(data[i+window_size, :month]) * 100 + Dates.month(data[i+window_size, :month])
end
for i in 1:test_size
   X_test[i, :] = [Dates.year(data[train_size + i + j - 1, :month]) * 100 + Dates.month(data[train_size + i + j - 1, :month]) for j in 1:window_size]
   y_test[i] = Dates.year(data[train_size + i + window_size, :month]) * 100 + Dates.month(data[train_size + i + window_size, :month])
end

# Normalize the input and output data
mean_x = mean(X_train, dims=1)
std_x = std(X_train, dims=1)
X_train = (X_train .- mean_x) ./ std_x
X_test = (X_test .- mean_x) ./ std_x

mean_y = mean(y_train)
std_y = std(y_train)
y_train = (y_train .- mean_y) ./ std_y
y_test = (y_test .- mean_y) ./ std_y

# Define the model
model = Chain(
   Dense(window_size, 10),
   Dense(10, 1)
)

# Define the loss function
loss_fn = Flux.MSE

# Define the optimizer
optimizer = Flux.ADAM

# Train the model
for epoch in 1:100
   Flux.train!(loss_fn, optimizer, Flux.params(model), X_train, y_train)
end

# Test the model
test_input = X_test[1:10, :]
test_output = model(test_input)
test_output = (test_output .* std_y) .+ mean_y
println("Prediction : ", test_output)
println("True value : ", y_test[1:10])