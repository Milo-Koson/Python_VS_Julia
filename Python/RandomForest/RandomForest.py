import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------ DATA PREPARATION ------------------------------------
features = pd.read_csv("Dataset/RandomForest/RandomForest.csv")

features = pd.get_dummies(features)
features.iloc[:, 5:].head(5)

labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
features = features.drop(['forecast_noaa', 'forecast_acc', 'forecast_under'], axis=1)
feature_list = list(features.columns)
features = np.array(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)

baseline_preds = test_features[:, feature_list.index('average')]
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# ------------------------------------ TRAINING ------------------------------------

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(train_features, train_labels)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)
# ------------------------------------ PREDICTION ------------------------------------

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

