
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('/content/AEP_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
df.head()

df.plot(style = '.',
        figsize = (15,5),
        color = 'blue',
        title = "PJME Energy Use in MW")
plt.show()

train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig,ax = plt.subplots(figsize = (15,5))
train.plot(ax = ax,label = "training set",title = 'Data Train')
test.plot(ax = ax,label = "test set")
ax.axvline('01-01-2015',color = 'black',ls = '--')
ax.legend(['training set','test set'])
plt.show()

df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')]\
.plot(figsize = (15,5),title = 'Week Of Data')
plt.show()

def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)


fig,ax = plt.subplots(figsize = (12,8))
sns.boxplot(data = df,x='hour',y='AEP_MW')
ax.set_title('MW by Hour')
plt.show()

fig,ax = plt.subplots(figsize = (12,8))
sns.boxplot(data = df,x='dayofweek',y='AEP_MW')
ax.set_title('MW by week')
plt.show()

df.columns

col = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear',
       'dayofmonth', 'weekofyear']
for i in col:
  fig,ax = plt.subplots(figsize = (12,8))
  sns.boxplot(data = df,x=i,y='AEP_MW')
  ax.set_title('MW by Hour')
  plt.show()

train = create_features(train)
test = create_features(test)

feature = col.drop('AEP_MW')
target = 'AEP_MW'

X_train = train[feature]
y_train = train[target]

X_test = test[feature]
y_test = test[target]

import xgboost as xgb

reg = xgb.XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    n_estimators=1000,
    early_stopping_rounds=50,
    objective='reg:squarederror',
    max_depth=5,
    learning_rate=0.01,
    colsample_bytree=0.8,
    subsample=0.8,
    gamma=0,
    min_child_weight=1,
    reg_alpha=0.1,
    reg_lambda=1,
    n_jobs=-1
)

reg.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True, suffixes=('', '_test'))
ax = df[['AEP_MW']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['AEP_MW'] \
    .plot(figsize=(15, 5), title='Week Of Data')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'] \
    .plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()

score = np.sqrt(mean_squared_error(y_test, test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')

test['error'] = np.abs(test[target] - test['prediction'])
test['date'] = test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)
