import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# to downlaod dataset
# !kaggle datasets download -d iabhishekofficial/mobile-price-classification

# !unzip /content/mobile-price-classification.zip

test = pd.read_csv('/content/test.csv')
train = pd.read_csv('/content/train.csv')

train.head()

train.columns

train.isnull().sum()

std = StandardScaler()

df = pd.concat([train,test],axis=0)
print(df)

df.isnull().sum()

df.drop('id',axis = 1,inplace = True)

df.dropna(inplace = True)

df.isnull().sum()

sns.pairplot(df)

x1 = df.drop('price_range',axis = 1)
y1 = df['price_range']

x1 = std.fit_transform(x1)

y1.value_counts()

pd.get_dummies(y1)

x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size = 0.2,random_state = 42)

model = Sequential()
model.add(Dense(128,activation = 'relu',input_shape = (x_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = 50,validation_data = (x_test,y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc = 'upper left')
