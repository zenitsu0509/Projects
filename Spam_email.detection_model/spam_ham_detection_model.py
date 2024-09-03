import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential

df = pd.read_csv("/content/spam.csv", encoding='latin-1')

df.head()

df_spam = df[df['v1']=='spam']
df_spam = df_spam[['v1','v2']]
df_spam

df_ham = df[df['v1'] == 'ham']
df_ham = df_ham[['v1','v2']]
df_ham

df_spam.shape

df_ham.shape

df_ham = df_ham.sample(df_spam.shape[0])

df_ham.shape

df_balance = pd.concat([df_spam,df_ham])

df_balance.reset_index(inplace=True)

df_balance

counts = df_balance['v1'].value_counts()
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
_ = plt.title('Distribution of Ham vs Spam')

df_balance.drop(['index'],axis=1,inplace=True)

df_balance['spam'] = df_balance['v1'].apply(lambda x: 1 if x == 'spam' else 0)
df_balance.head()

df_balance

X_train, X_test, y_train, y_test = train_test_split(df_balance['v2'],df_balance['spam'],test_size=0.2,random_state = 42)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

X_test_vec

model = LogisticRegression()
model.fit(X_train_vec, y_train)

model1 = MultinomialNB()
model1.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

y_pred_1 = model1.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred_1)
rep = classification_report(y_test, y_pred_1)

print(f"Accuracy: {acc}")
print(f"Classification Report:\n{rep}")
confusion_matrix(y_test, y_pred_1)

model.predict(vectorizer.transform(["With reference to NSE circulars NSE/INSP/46704 dated December 17, 2020, and NSE/INSP/55039 e to NCL securities ISIN wise is attached"]))

joblib.dump(model1, 'spam_model_nb.joblib')

joblib.dump(model, 'spam_model_lr.joblib')

model1.predict(vectorizer.transform(["Hello,Above is the OTP to process your account closure request. Do not share your OTP with anyone for security reasons.	 "]))

model = Sequential()
model.add(Dense(64,activation='relu',input_shape=(X_train_vec.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation = "sigmoid"))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train_vec,y_train,epochs =10)

model.evaluate(X_test_vec,y_test)

y = model.predict(vectorizer.transform(["After launching an experimental updated version of Gemini 1.5 Pro (0801) that ranked #1 on the LMSYS leaderboard, we’re introducing a series of improvements to help developers scale with our most capable and efficient models"]))

if y > 0.5:
  print("spam")
else:
  print("ham")

