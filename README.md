# ESEO PFE Python vs Julia

## Team:
- Milo Koson (Developer)
- Kevin Sen (Developer)
- Louison Tendron (Developer)

## Objective:
Compare coding language Python versus Julia, in the context of data manipulation and machine learning applied to time
series

## Tasks:
1. Develop standard Python code for time series prediction (one using ARIMA, STL-Arima, Random Forest,
XGBoost, Keras NN / LSTM / CNN) and recode the same in Julia, using only Julia native function as much as
possible (no wrappers).
You can rely on https://machinelearningmastery.com/ for tutorials in Python, and rely on https://julialang.org/
for Julia to start with.
2. Implement a method to quantitative and qualitative compare the two. Compare capabilities, performance, ease
of use, readability, code size …
3. Document the Python and Julia resources used side-by-side (Web site links for tutorials, libraries …).
4. Document the setup to develop run the entire tests, so some can follow your step by step install procedure to
reproduce your results

## Features to Test:
- Files/Database connection (S3, delta.io/spark, Clickhouse)
  - read, write
  - Load data
  - Build and execute SQL and/or code in delta.io/spark, Clickhouse
- Data manipulation
  - Scaling, normalization
  - Encoding : Ordinal / Label, On-Hot, Binary,
  - Shift (data preparation for time series)
  - Split Train / Test
- Perform time series predictions
  - ARIMA / SARIMA / SARIMAX statsmodels.tsa.statespace.sarimax.SARIMAX - statsmodels 0.14.0
  - Equivalent to STL or MSTL Forecast with Arima (STL = Season-Trend decomposition using LOESS)
  https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.stl.STLForecast.html
  - Regressors like Random Forest or XGBoost https://scikitlearn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html ,
  https://xgboost.readthedocs.io/en/latest/python/python_intro.html
  - Neural Network, Equivalent to Keras Embeding, Dense, LSTM, CNN…
- Use docker as much as possible to deploy the environments, under linux (you can use the Windows Subsystem
  for Linux (WSL) if you have recent windows.)
- If possible check also unsupervised learning models for anomalies detection and classification use case
  Note that Thales if needed can provide a dataset, but to not lose time you can also choose to start with the dataset
  found within online tutorials as long as you are using a matching dataset for a test of the two language to be able to
  compare.

## Expected delivery:
- All the python's models are located in the "Python" repository.
- All the Julia's models are located in the "Julia" repository.
- The final comparison document, the Excel summary file, the poster and our PowerPoint presentation
are located in the "Deliverables" repository.
