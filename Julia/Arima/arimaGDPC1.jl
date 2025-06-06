# https://lost-stats.github.io/Time_Series/ARIMA-models.html
# https://lampspuc.github.io/StateSpaceModels.jl/latest/manual/

# Load necessary packages
using StateSpaceModels, CSV, Dates, DataFrames, LinearAlgebra, Plots

# Import data
data = CSV.File("Dataset/Arima/GDPC1.csv") |> DataFrame

# Specify GDPC1 series as an ARIMA(2,2,2) model
model = SARIMA(data.GDPC1, order = (2,2,2))

# Fit (estimate) the model
fit!(model)

# Print estimates
print_results(model)

forec = forecast(model, 24)
plot(model, forec; legend = :topleft, title="ARIMA Prediction", label=["Original Series" "Prediction"], lindewidth=3)