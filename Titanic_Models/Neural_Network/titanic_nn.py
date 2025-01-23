import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np

df = pd.read_csv('/content/titanic2.csv')
df.head()

df.isnull().sum()

df['Embarked'] = df['Embarked'].fillna('S')

df['Embarked'].replace({'S':1, 'C':2, 'Q':3}, inplace=True)

mean = df['Age'].mean()
df['Age'].fillna(mean, inplace=True)

df['Sex'].replace({'male':0, 'female':1}, inplace=True)

df  = df.drop(['Name','PassengerId','Ticket','Cabin'],axis = 1)

df.isnull().any()
print(df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit_transform(df)

df.shape

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop('Survived',axis = 1),df['Survived'],test_size = 0.2,random_state = 42)

inputs = keras.Input(shape=(X_train.shape[1],))
x = layers.BatchNormalization()(inputs)
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
histoy = model.fit(X_train,y_train,epochs = 50,batch_size = 32,verbose = 2)
loss, accuracy = model.evaluate(X_test, y_test, verbose=2, batch_size=32)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model)
