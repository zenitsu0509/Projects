import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('/content/sample_data/california_housing_train.csv')
test = pd.read_csv('/content/sample_data/california_housing_test.csv')

train


from matplotlib import pyplot as plt
train.plot(kind='scatter', x='longitude', y='latitude', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)



from matplotlib import pyplot as plt
train.plot(kind='scatter', x='total_rooms', y='total_bedrooms', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

std = StandardScaler()
train = std.fit_transform(train)
test = std.fit_transform(test)

train_df = pd.DataFrame(train, columns=['longitude', 'latitude', 'housing_median_age',
                                       'total_rooms', 'total_bedrooms', 'population',
                                       'households', 'median_income', 'median_house_value'])
test_df = pd.DataFrame(test, columns=['longitude', 'latitude', 'housing_median_age',
                                      'total_rooms', 'total_bedrooms', 'population',
                                      'households', 'median_income', 'median_house_value'])
X_train = train_df.drop('median_house_value', axis=1)
y_train = train_df['median_house_value']
X_test = test_df.drop('median_house_value', axis=1)
y_test = test_df['median_house_value']

model = RandomForestRegressor(n_estimators=200,max_depth = None,max_features='log2')
model.fit(X_train, y_train)

model.score(X_test, y_test)

model2 = GradientBoostingRegressor(n_estimators=200,max_depth= 10)
model2.fit(X_train, y_train)

model2.score(X_test, y_test)

mean_squared_error(y_test, model.predict(X_test))

y_pred = model.predict(X_test)
top_5_pred = y_pred[:5]
top_5_true = y_test[:5]

print("\nTop 5 Predictions vs Original Values:")
for i in range(5):
    adjusted_prediction = top_5_pred[i] * (np.std(y_pred) + np.mean(y_pred))
    original_value = top_5_true[i] * (np.std(y_test) + np.mean(y_test))
    print(f"Prediction: {adjusted_prediction:.2f}, Original: {original_value}")

mean_squared_error(y_test, model2.predict(X_test))

y_pred = model2.predict(X_test)
top_5_pred = y_pred[:5]
top_5_true = y_test[:5]
print("\nTop 5 Predictions vs Original Values:")
for i in range(5):
    adjusted_prediction = top_5_pred[i] * (np.std(y_test) + np.mean(y_test))
    original_value = top_5_true[i] * (np.std(y_test) + np.mean(y_test))
    print(f"Prediction: {adjusted_prediction:.2f}, Original: {original_value}")