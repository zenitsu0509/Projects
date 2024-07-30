import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import pandas as pd
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('/content/sample_data/california_housing_train.csv')
test = pd.read_csv('/content/sample_data/california_housing_test.csv')

train

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

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(8,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2,verbose = 2)

y_pred = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)  # Get the actual predictions
y_pred = y_pred.flatten() # Flatten the predictions to a 1D array

top_5_pred = y_pred[:5]
top_5_true = y_test[:5]

print("\nTop 5 Predictions vs Original Values:")
for i in range(5):
    print(f"Prediction: {top_5_pred[i]:.2f}, Original: {top_5_true[i]:.2f}")

