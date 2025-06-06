# https://github.com/LAMPSPUC/StateSpaceModels.jl/blob/master/docs/src/examples.md

using CSV, DataFrames, StateSpaceModels, Plots

# Getting the data 
air_passengers = CSV.File("DataSet/Arima/airPassengers.csv") |> DataFrame

# Converting data for the plotting
log_air_passengers = log.(air_passengers.passengers)

# Plotting the data 
plot(log_air_passengers, title="Original Series", lindewidth=3)

# Applying SARIMA (Seasonal + ARIMA) to our data
model = SARIMA(log_air_passengers; order = (0, 1, 1), seasonal_order = (0, 1, 1, 12))
fit!(model)
print_results(model)

# Getting our forecast on our SARIMA Model
forec = forecast(model, 24)

# Plotting the forecasts results
plot(model, forec; legend = :topleft, title="SARIMA Prediction", label=["Original Series" "Prediction"], lindewidth=3)

