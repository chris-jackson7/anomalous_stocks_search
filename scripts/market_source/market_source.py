import numpy as np
import pandas as pd
import requests
import os
import sys
import pickle
from datetime import datetime
import time

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import xgboost as xgb

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

from ..constants import (
    LINKS, OBJECT_EXCLUDE_COLS, CATEGORICAL_DROP_COLS,
    MODEL_COLS, ROW_MISSING_THRESHOLD, TARGET_COLUMN
)


# saving time of last run
with open('./assets/last_updated.txt', 'w') as file:
    file.write(datetime.now().strftime("%Y-%m-%d"))

##### COLLECTING DATA FROM API #####


def make_request(url: str, first: bool = False) -> pd.DataFrame:

    json = requests.get(url).json()
    try:
        result = pd.DataFrame(json['data']['data'])
    except:
        result = pd.DataFrame(json)
    result.index = result.iloc[:,0]

    if first:
        #result.drop('s', inplace=True, axis=1)
        pass
    else:
        result.drop(0, inplace=True, axis=1)
        if 2 in result.columns:
            result.drop(2, inplace=True, axis=1)
        result.columns = [url.split('/')[-1]]
    return result


def merge_data(dfs):
    for i, df in enumerate(dfs):
        if i == 0:
            orig_df = df
            continue
        orig_df = orig_df.join(df, how='outer')
    return orig_df


URL = 'https://stockanalysis.com/api/screener/s/i'


dfs = [make_request(URL, True)]
dfs.extend([make_request(link['link']) for link in LINKS])
df = merge_data(dfs)


##### SAVING PRICE & VOLUME DATA #####


def update_time_data(column: str) -> None:
    file_name = f"{column}_over_time.pkl"

    if file_name in os.listdir():
        with open(file_name, 'rb') as file:
            historical_data = pickle.load(file)

        updated_data = pd.merge(historical_data, df[['s', column]], on='s')
        pickle.dump(updated_data, open(file_name, 'wb'))
    else:
        pickle.dump(df[['s', column]], open(file_name, 'wb'))

update_time_data('price')
update_time_data('volume')


##### CLEANING & IMPUTING RAW DATA #####


df.infer_objects()

# Handling Dates
object_columns_mask = np.array(df.dtypes) == 'object'
object_date_columns = [col for col in df.columns[object_columns_mask] if 'Date' in col]

def epoch_time_helper(date):
  try:
    return datetime.strptime(date, '%Y-%m-%d').timestamp()
  except:
    return None

for col in object_date_columns:
  df[col] = df[col].apply(epoch_time_helper)
  df.loc[df[col].isna(), col] = 0
  df[col] = df[col].astype(int)


# Transforming Floats
missing_rates = df.isna().mean()
float_columns_mask = (np.array(df.dtypes) == 'float64') & (np.array(missing_rates) < .4)
float_cols = [col for col in df.columns[float_columns_mask]]
try:
    print(f"excluded float cols:\n{df.columns[(np.array(df.dtypes) == 'float64') & ~float_columns_mask]}")
except Exception as e:
    print(e)

for col in float_cols:
  if any(df[col] == 0):
    # jitter for log transformation
    df.loc[df[col] == 0, col] = .0001
  df[f'log_{col}'] = df[col].apply(np.log)
df['asinh_roic'] = df['roic'].apply(np.arcsinh)

float_cols.extend([f'log_{col}' for col in float_cols])
float_cols.append('asinh_roic')


# Standard Scaling Floats
scaler = StandardScaler()
scaler.fit(df.loc[:, float_cols])
df.loc[:, float_cols] = scaler.transform(df.loc[:, float_cols])


# Reassigning Sector Values
df.loc[df['sector'] == 'Equity Real Estate Investment Trusts (REITs)', 'sector'] = 'Real Estate'
df.loc[df['sector'] == 'Finance', 'sector'] = 'Financials'
df.loc[df['sector'] == 'Independent Power and Renewable Electricity Producers', 'sector'] = 'Energy'
df.loc[df['sector'] == 'Health Services', 'sector'] = 'Healthcare'
df.loc[df['sector'].isna(), 'sector'] = 'Miscellaneous'
df.loc[df['sector'] == 'Distribution Services', 'sector'] = 'Miscellaneous'
df.loc[df['sector'] == 'Blank Check / SPAC', 'sector'] = 'Miscellaneous'

industry_sector = df[['industry', 'sector']]


# One Hot Encoding Sector
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
sector_reshaped = np.array(df["sector"]).reshape(-1, 1)
encoder.fit(sector_reshaped)
encoded_sector = pd.DataFrame(
    encoder.transform(sector_reshaped),
    columns=encoder.get_feature_names_out()
)
encoded_sector.columns = [col.split('_')[1] for col in encoded_sector.columns]

# Plot Sector Distribution
sector_dist = encoded_sector.sum(axis=0).sort_values()
plt.barh(sector_dist.index, sector_dist, color='skyblue')
plt.title('Sectors Distribution')
plt.savefig('./assets/sector_distribution.png')

encoded_sector.index = df.index
df = pd.concat([df, encoded_sector], axis=1)


# Removing Chinese Companies
df = df[df['country'] != 'China']

# df.drop('price', axis=1, inplace=True)
# df.drop(CATEGORICAL_DROP_COLS, axis=1, inplace=True)


# Saving Cleaned and Transformed Data
print(f'data size: {sys.getsizeof(df)//1e6}MB')
pickle.dump(df, open(f'market_data_transformed.pkl', 'wb'))


# Drop Missing Rows for Lowest Missing Rate Columns & Target
print(df.shape)
df = df.dropna(subset = ['ch1m', 'ch6m', TARGET_COLUMN]) # log_price
print(df.shape)

# Include Target & Sector Encodings in Model Columns
MODEL_COLS.append(TARGET_COLUMN)
MODEL_COLS.extend(list(encoded_sector.columns))
df = df[MODEL_COLS]


# Plot Row Missing Rate Distibution
rows_missing_rate = df.isnull().mean(axis=1)
sns.kdeplot(rows_missing_rate)
plt.title('Row Missing Rate Distribution')
plt.savefig('./assets/row_missing_rate.png')

# Drop Rows w/ High Missing Rate
df = df[rows_missing_rate < ROW_MISSING_THRESHOLD]
symbols = df.index
print(df.shape)


# Impute Remaining Missing Data
imputer = KNNImputer(n_neighbors=15)
df = pd.DataFrame(
    imputer.fit_transform(df),
    columns = MODEL_COLS,
    index=symbols
)


##### MODEL TRAINING & EVALUATION #####


# Train Test Split Data
y = df[TARGET_COLUMN]
X = df.drop(TARGET_COLUMN, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)


# Define Model Parameters
params = {
    'n_estimators': 100,  # Number of trees
    'max_depth': 6,  # Maximum depth of trees
    'learning_rate': 0.1,  # Learning rate
    'subsample': 0.8,  # Subsample ratio for each tree
    'colsample_bytree': 0.8,  # Feature subsample ratio per tree
    'objective': 'reg:squarederror',  # Objective function for regression (change for classification)
    'eval_metric': 'rmse',  # Evaluation metric for regression (change for classification)
    'random_state': 42,  # Set random seed for reproducibility
    'booster': 'gblinear' # IMPORTANT: only allow linear relationships
}

# Create an XGBoost Random Forest Model
model = xgb.XGBRegressor(**params)  # Use XGBRegressor for regression, XGBRFRegressor for classification

# Train the Model
model.fit(X_train, y_train, eval_set=[(X_test, y_test)]) # , early_stopping_rounds=10

# Extract Evaluation
rmse = min(model.evals_result()['validation_0']['rmse'])
with open('assets/model_rmse.txt', 'w') as file:
   file.write(str(rmse))

# Make Predictions
y_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)


# Visualizing Train / Test Fits & Feature Importance
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 7))

axes[0].scatter(y_train, y_train_pred)
axes[0].set_title('Prediction on Training Data')
axes[0].axline([0, 0], slope=1, color='red', linestyle='--')
axes[0].grid()

axes[1].scatter(y_test, y_pred)
axes[1].set_title('Prediction on Test Data')
axes[1].axline([0, 0], slope=1, color='red', linestyle='--')
axes[1].grid()

# Extract Feature Importance
feature_importance = sorted(model.feature_importances_)
threshold = min(feature_importance)
feature_importance = np.array(feature_importance)[feature_importance >= threshold]
important_features = np.where(feature_importance >= threshold)[0]

features, importance = zip(*sorted(
    [(x,y) for x,y in zip(
        feature_importance,
        np.array(X.columns)[important_features].tolist()
        )
    ], key=lambda x:x[0], reverse=False)
)

axes[2].barh(importance, features, color='skyblue')
axes[2].set_title('Feature Importance')
axes[2].grid(axis='x', linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.savefig('./assets/train_test_fit_feature_importance.png')


def linear_fit_and_se(x, y):
    m, b = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
    y_pred = m * x + b
    squared_residuals = np.square(y - y_pred)
    return squared_residuals

# combining train and test data
index = list(y_test.index)
index.extend(list(y_train.index))
y = list(y_test)
y.extend(y_train)
pred = list(y_pred)
pred.extend(y_train_pred)

search_df = pd.DataFrame({'y': y, 'pred': pred}, index=index)

# Standard Errors
se = linear_fit_and_se(search_df['y'], search_df['pred'])

# Isolation Forest
clf = IsolationForest(n_estimators=100, max_samples="auto", bootstrap=True)
clf.fit(search_df)
iso_score = clf.decision_function(search_df)
iso_score = iso_score - min(iso_score) + .001

# 2D Z Scores
z_score = np.linalg.norm(search_df[['y', 'pred']], axis=1)

anomaly_scores_df = pd.DataFrame({
    'se': se / max(se),
    'iso': (max(iso_score) - iso_score) / max(iso_score), # reversing distribution to more closely match z_score and se
    'z_score': z_score / max(z_score)
    # scaled all to (0, 1)
})

search_df = pd.concat([search_df, anomaly_scores_df], axis=1)

# Plotting Anomaly Distributions
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
sns.kdeplot(anomaly_scores_df['z_score'], ax=axes[0])
sns.kdeplot(anomaly_scores_df['iso'], ax=axes[1])
sns.kdeplot(anomaly_scores_df['se'], ax=axes[2])
axes[0].set_title('Z Score')
axes[1].set_title('Isolation Score')
axes[2].set_title('Squared Error')
plt.savefig('./assets/anomaly_distributions.png')


# Add Sector & Industry Back
df_temp = pd.merge(
    X.round(4),
    industry_sector,
    left_index=True, right_index=True
)

search_data = pd.merge(
    search_df.round(4),
    df_temp.reindex(columns=df_temp.columns[::-1]),
    left_index=True, right_index=True
)

pickle.dump(search_data, open(f'search_df.pkl', 'wb'))
