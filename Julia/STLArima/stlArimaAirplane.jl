using CSV, Plots, SeasonalTrendLoess, DataFrames, StateSpaceModels

# Load the dataset
df = CSV.File("Dataset/Arima/airPassengers.csv") |> DataFrame

# Extract the column containing the number of passengers
passengers = log.(df.passengers)

# Apply the STL decomposition
stl_result = stl(passengers, 12; robust=true)

# Plot the original data and the STL decomposition
plot(df.month, passengers, label="Original Data", xlabel="Month", ylabel="Number of Passengers", linecolor=:blue)
plot!(df.month, stl_result.trend, label="Trend Component", linecolor=:red)
plot!(df.month, stl_result.seasonal, label="Seasonal Component", linecolor=:green)
plot!(df.month, stl_result.remainder, label="Remainder Component", linecolor=:purple)


# Prédiction avec ARIMA sur la composante résiduelle
model = SARIMA(stl_result.remainder, order = (0, 1, 1), seasonal_order = (0, 1, 1, 12))
fit!(model)
forec = forecast(model, 24)


plot(model, forec; legend = :topleft, title="SARIMA Prediction", label=["Original Series" "Prediction"], lindewidth=3)

