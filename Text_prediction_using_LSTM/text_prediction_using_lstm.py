from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt
import sys

filename = "/content/pg236.txt"
raw_text = open(filename,'r',encoding='utf-8').read()
print(raw_text[:1000])

raw_text = ' '.join(c for c in raw_text if not c.isdigit())
chars = sorted(list(set(raw_text)))
print(chars)

char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))

n_chars = len(raw_text)
v_vocab = len(chars)
print("Total Characters: ",n_chars)
print("Total Vocab: ",v_vocab)

seq_length = 80
step = 8
sentences = []
next_char = []
for i in range(0, len(raw_text) - seq_length, step):
    sentences.append(raw_text[i: i + seq_length])
    next_char.append(raw_text[i + seq_length])
n_pattens = len(sentences)
print("Total Patterns: ", n_pattens)

sentences[0:5]

x = np.zeros((len(sentences),seq_length,v_vocab),dtype=bool)
y = np.zeros((len(sentences),v_vocab),dtype=bool)
for i,sentence in enumerate(sentences):
    for t,char in enumerate(sentence):
        x[i,t,char_to_int[char]] = 1
    y[i,char_to_int[next_char[i]]] = 1
# summarize the loaded data
print(x.shape)
print(y.shape)
print(y[0:5])

model = Sequential()
model.add(LSTM(128,input_shape=(seq_length,v_vocab)))
model.add(Dense(v_vocab,activation='softmax'))
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)

model.summary()

from keras.callbacks import ModelCheckpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list = [checkpoint]

history = model.fit(x,y,batch_size=128,epochs=50,callbacks = callbacks_list)

model.save_weights("model.h5")

loss = history.history['loss']
epochs = range(1,len(loss) +1)
plt.plot(epochs,loss,'y',label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

def sample(preds):
  preds = np.array(preds).astype('float64')
  preds = np.log(preds)
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1,preds,1)
  return np.argmax(probas)

filename = "/content/model.h5"
model.load_weights(filename)
start_index = np.random.randint(0,len(raw_text) - seq_length -1)
generated = ''
sentence = raw_text[start_index:start_index + seq_length]
generated += sentence
print('Generating with seed: "' + sentence + '"')
sys.stdout.write(generated)

for i in range(400):
  x_pred = np.zeros((1,seq_length,v_vocab))
  for t,char in enumerate(sentence):
    x_pred[0,t,char_to_int[char]] = 1
  preds = model.predict(x_pred,verbose=0)[0]
  next_index = sample(preds)
  next_char = int_to_char[next_index]
  generated += next_char
  sentence = sentence[1:] + next_char
  sys.stdout.write(next_char)
  sys.stdout.flush()
print()

