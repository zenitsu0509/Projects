import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/content/titanic2.csv")
df[0:10]

df.isnull().sum()

df.columns

df.drop(['Cabin','Ticket','Name','PassengerId'],axis = 1,inplace = True)

sns.pairplot(df)
plt.show()

for col in df.columns:
  if col != 'Survived':
    plt.figure(figsize=(8,4))
    sns.countplot(x=col, hue="Survived", data=df)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.title(f"{col} vs Survived")
    plt.show()

sns.distplot(df.Age)

def impute_fill(df,col,mean):
  df[col +'_mean'] = df[col].fillna(mean)
  return df
age_mean = df.Age.mean()
age = impute_fill(df,'Age',age_mean)

df.drop(['Age'],axis = 1,inplace= True)

df

df = df.dropna(axis = 0)

df = pd.get_dummies(data=df,drop_first=True)
df.info()

df.isnull().sum()

X = df.drop(['Survived','Embarked_S'],axis = 1)
y = df['Survived']
print(X)
print(y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=43,test_size=.30)

lg = LogisticRegression(max_iter=100)
# X_train_reshaped = X_train.values.reshape(-1,1)
lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)
acc = lg.score(X_test,y_test)
acc_pred = sum(yt == yp for yt,yp in zip(y_test,y_pred))
acc_pred = acc_pred/len(y_test)
print(acc_pred)

gbc =GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred = gbc.predict(X_test)
acc_pred = sum(yt == yp for yt,yp in zip(y_test,y_pred))
print(acc_pred/len(y_test))

abc = AdaBoostClassifier()
abc.fit(X_train,y_train)
y_pred = abc.predict(X_test)
acc_pred = sum(yt == yp for yt,yp in zip(y_test,y_pred))
print(acc_pred/len(y_test))

xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
acc_pred = sum(yt == yp for yt,yp in zip(y_test,y_pred))
print(acc_pred/len(y_test))

