import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

df = pd.read_csv("/content/nifty50 data.csv")
data = df

df.dropna(inplace = True)

df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()

plt.figure(figsize = (16,8))
plt.title("Close Price Visualization")
plt.plot(df.Close)

moving_100 = df['Close'].rolling(100).mean()

plt.figure(figsize = (16,8))
plt.title("Close Price Visualization")
plt.plot(df['Close'])
plt.plot(moving_100, 'r')

train = pd.DataFrame(data[0:int(len(data)*0.70)])
test = pd.DataFrame(data[int(len(data)*0.70): int(len(data))])

print(train.shape)
print(test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values

data_training_array = scaler.fit_transform(train_close)
data_test_array = scaler.transform(test_close)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_test = []
y_test = []
for i in range(100, data_test_array.shape[0]):
    x_test.append(data_test_array[i-100: i])
    y_test.append(data_test_array[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

x_train.shape

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences=True
              ,input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))


model.add(LSTM(units = 60, activation = 'relu', return_sequences=True))
model.add(Dropout(0.3))


model.add(LSTM(units = 80, activation = 'relu', return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

print("NaN in train_scaled:", np.isnan(x_train).sum())
print("Infinite in train_scaled:", np.isinf(x_train).sum())
print("NaN in test_scaled:", np.isnan(x_test).sum())
print("Infinite in test_scaled:", np.isinf(x_test).sum())

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['MAE'])
model.fit(x_train, y_train, validation_data = (x_test, y_test) ,epochs = 50)

import tensorflow as tf
tf.keras.models.save_model(model, 'my_model.keras')

past_100_days = pd.DataFrame(train_close[-100:])

test_df = pd.DataFrame(test_close)

import pandas as pd
final_df = pd.concat([past_100_days, test_df], ignore_index=True)

input_data = scaler.fit_transform(final_df)
input_data

y_pred = model.predict(x_test)

scale_factor = 1/0.00985902
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error on test set: ", mae)

