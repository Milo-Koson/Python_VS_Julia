using CSV, DataFrames, Dates, Flux, Statistics, MLUtils

# Loading CSV file
df = CSV.File("Dataset/CNN/airPassengers.csv") |> DataFrame

# Convert 'month' column in Date type
df.month = Dates.Date.(string.(df.month), "yyyy-mm-dd")

# Extract year and month as numerical features
df.year = Dates.year.(df.month)
df.month = Dates.month.(df.month)

# Replace months by a sequence of numbers from 1 to the length of the months column
df.month = 1:length(df.month)

# Extracting time sequences
sequences = [Float32.(df.passengers)]

# Splitting training data and test data
train_data, test_data = Flux.splitobs(sequences[1], at=0.8)

# Defining the time "window" for the CNN 1D
window_size = 12

# Preparing the time series data
X = []
y = []
for i in 1:length(train_data) - window_size
    push!(X, train_data[i:i+window_size-1])
    push!(y, train_data[i+window_size])
end

# Reshaping X and y to add a dimension
X = reduce(hcat, X)'
X = reshape(X, (:,window_size, 1))
y = reshape(y, (length(y), 1))

# Creating the dataloader for (X,y)
data_loader = DataLoader((X, y), batchsize=10, shuffle = true)

# Defining the architecture for the model
input_size = (window_size, 1)
num_classes = 1

# Defining the 1D CNN architecture
function create_cnn(input_size, num_classes)
    return Chain(
        Conv((2, 1), 1=>16, relu),
        MaxPool((2, 1)),
        Dense(16, num_classes),
    )
end

cnn_model = create_cnn(input_size, num_classes)

# Defining loss function and the optimizer
loss(x, y) = Flux.mse(cnn_model(x), y)
optimizer = ADAM(0.001)

# Training the model
epochs = 100
for epoch in 1:epochs
    for (X, y) in data_loader
        X = cat(X..., dims=2)
        y = cat(y..., dims=2)
        Flux.train!(loss, Flux.params(cnn_model), [(X, y)], optimizer)
    end
    println("Epoch: $epoch, Loss: $(loss(X, y))")
    println("Model weights: ", [params(cnn_model)[i] for i in 1:length(params(cnn_model))])
    println("Loss on entire training data: $(loss(X, y))")
end

# Evaluate the model
test_X = cat(test_data..., dims=2)[1:end-window_size, :]
test_y = cat(test_data..., dims=2)[window_size+1:end, :]

mse = Flux.mse(cnn_model(test_X), test_y)

println("Mean Squared Error on Test Data: $mse")