using MLJ, DecisionTree, CSV, DataFrames

# Load data
data = CSV.File("Dataset/RandomForest/RandomForest.csv") |> DataFrame
labels = convert(Array, data[!, :actual])

# Remove redundant columns
select!(data, Not([:actual, :forecast_noaa, :forecast_acc, :forecast_under]))

# Convert categorical columns to CategoricalArray types
categorical_columns = [:year, :month, :day, :week, :temp_2, :temp_1, :average, :friend]
for col in categorical_columns
    data[!, col] = categorical(data[!, col])
end

# One-hot encoding of categorical columns
hot = OneHotEncoder()
mach = machine(hot, data[:, categorical_columns])
MLJ.fit!(mach)
new_data = MLJ.transform(mach, data)

# Convert the transformed data to DataFrame
new_data = DataFrame(new_data)

# Convert all columns to Float64
for col in names(new_data)
    new_data[!, col] = convert(Vector{Float64}, new_data[!, col])
end

# Remove original categorical columns
non_categorical_columns = setdiff(names(data), categorical_columns)
new_data = hcat(new_data, data[:, non_categorical_columns])

# Ensure labels are also numerical
labels = convert(Vector{Float64}, labels)

# Define the model
model = @load RandomForestClassifier pkg=DecisionTree

# Split the new_data into training and test sets
train, test = partition(eachindex(labels), 0.7, shuffle=true) 

model_instance = model()

# Wrap the model in machine with new_data
mach = machine(model_instance, new_data, labels)

# Train the model
MLJ.fit!(mach, rows=train)

# Predict on the test set
yhat = predict(mach, rows=test)

# Measure the performance
accuracy = mean(yhat .== labels[test])
println("Accuracy: $accuracy")