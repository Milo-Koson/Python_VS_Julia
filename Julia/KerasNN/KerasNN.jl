using CSV, DataFrames, Dates, Flux, Plots

# Charger les données à partir du fichier CSV
data = CSV.read("Dataset/KerasNN/airPassengers.csv", DataFrame)

# Convert 'month' column in Date type
data.month = Dates.Date.(string.(data.month), "yyyy-mm-dd")

# Extract year and month as numerical features
data.year = Dates.year.(data.month)
data.month = Dates.month.(data.month)

# Replace months by a sequence of numbers from 1 to the length of the months column
data.month = 1:length(data.month)

# Extracting time sequences
sequences = [Float32.(data.passengers)]

# Splitting training data and test data
train_size = floor(Int, size(data, 1) * 0.8)
train_data = data[1:train_size, :]
test_data = data[train_size+1:end, :]

# Building a model with 2 dense layers
model = Chain(
    Dense(1, 10, relu),
    Dense(10, 1)
)

# Prepare training data
X_train = transpose(Float32.(train_data.month))
Y_train = transpose(Float32.(train_data.passengers))

X_test = transpose(Float32.(test_data.month))
Y_test = transpose(Float32.(test_data.passengers))

# Defining loss function and optimizer
loss(x, y) = Flux.mse(model(x), y)
optimizer = ADAM()

# Training the model
Flux.train!(loss, Flux.params(model), [(X_train, Y_train)], optimizer)

# Model evaluation
test_loss = loss(X_test, Y_test)
println("Test Loss: $test_loss")

# Make predictions
predictions = model(X_test)

# Convert Y_test and predictions to vectors
Y_test_vec = vec(Y_test)
predictions_vec = vec(predictions)

# Tracing predictions against actual data
plot(test_data.month, Y_test_vec, label="Actual")
plot!(test_data.month, predictions_vec, label="Predicted")