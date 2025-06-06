using XGBoost, DataFrames, CSV, Dates, Plots, MLLabelUtils

# Loading the Dataset
df = CSV.File("Dataset/XGBoost/airPassengers.csv") |> DataFrame

# Converting 'month' column in Date type
df.month = Dates.Date.(string.(df.month), "yyyy-mm-dd")

# Conserver 'month' sous forme de dates
df.month = Dates.value.(df.month)

# Convert DataFrame to matrices 
X = select(df, Not(:passengers))
y = df[!, :passengers]

# Define the model parameters
param = Dict(
    "objective" => "reg:squarederror",
    "max_depth" => 6,            # Tree maximum depth
    "eta" => 0.1,                # Learning rate
    "subsample" => 0.8,          # Fraction of samples used to train each tree
    "colsample_bytree" => 0.8    # Fraction of features used to train each tree
)

# Train the xgboost model
dtrain = DMatrix(X, label=y)
model = xgboost(dtrain, num_round=15, param=param)

# Making predictions with xgboost model
predictions = predict(model, dtrain)

println("Prédictions: ", predictions)

# Ploting the results
plot(df.month, y, label="Observations", xlabel="Date", ylabel="Passengers", legend=:top)
plot!(df.month, predictions, label="Prédictions", color=:red)
