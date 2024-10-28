
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df = pd.read_csv("/content/nifty50 data.csv")

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

train = df[0:int(len(df)*0.70)]
test = df[int(len(df)*0.70): int(len(df))]

print(train.shape)
print(test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values

data_training_array = scaler.fit_transform(train_close)

x_train = []
y_train = []
time_steps = 100

for i in range(time_steps, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# x_test = []
# y_test = []
# for i in range(time_steps, data_test_array.shape[0]):
#     x_test.append(data_test_array[i-100: i])
#     y_test.append(data_test_array[i, 0])

# x_test, y_test = np.array(x_test), np.array(y_test)

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
# print("NaN in test_scaled:", np.isnan(x_test).sum())
# print("Infinite in test_scaled:", np.isinf(x_test).sum())

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['MAE'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

model.fit(x_train, y_train, epochs=30, batch_size=32) #, callbacks=[early_stopping, lr_scheduler]

import tensorflow as tf
tf.keras.models.save_model(model, 'my_model.keras')

past_100_days = train_close[-100:]

final_df = pd.concat([pd.DataFrame(past_100_days), pd.DataFrame(test_close)], ignore_index=True)

input_data = scaler.fit_transform(final_df)
x_final_test = [input_data[i-100:i] for i in range(100, input_data.shape[0])]
x_final_test = np.array(x_final_test)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
   x_test.append(input_data[i-100: i])
   y_test.append(input_data[i, 0])

y_pred = model.predict(x_final_test)

scaler.scale_

scale_factor = 1/9.93936984e-07
y_pred = y_pred * scale_factor
y_test = np.array(y_test) * scale_factor

y_pred = y_pred.reshape(-1)
plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("Mean absolute error on test set: ", mae)

