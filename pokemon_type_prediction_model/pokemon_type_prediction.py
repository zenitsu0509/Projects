import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/images')

images = os.listdir()
print(images)

from PIL import Image
import matplotlib.pyplot as plt
for image_path in images[0:5]:
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def process_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

image_size = (224, 224)
image_dir = '/content/drive/MyDrive/images'

label_file_path = '/content/pokemon.csv'
labels_df = pd.read_csv(label_file_path)
labels_df.head()

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
# labels_df['Type2'].fillna('None', inplace=True)
# labels_df['Evolution'].fillna('None', inplace=True)
encoded_labels = encoder.fit_transform(labels_df[['Type1']])
encoded_labels_df = pd.DataFrame(encoded_labels, columns=encoder.get_feature_names_out(['Type1']))
encoded_labels_df.insert(0, 'Name', labels_df['Name'])
encoded_labels_df.head()

type1_columns = [col for col in encoded_labels_df.columns if col.startswith('Type1_')]
type1_labels = encoded_labels_df[type1_columns]
type1_labels.head()

labels_df = pd.read_csv('/content/pokemon.csv')
print(labels_df.head())

csv_file = '/content/pokemon.csv'
df = pd.read_csv(csv_file)
image_dir = '/content/drive/MyDrive/images'
img_size = (128, 128)
name_to_type1 = dict(zip(df['Name'].str.lower(), df['Type1']))
images = []
labels = []

for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        name = os.path.splitext(filename)[0].lower()

        if name in name_to_type1:

            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path).resize(img_size)
            image = img_to_array(image) / 255.0
            images.append(image)
            labels.append(name_to_type1[name])

images = np.array(images)
labels = np.array(labels)

print(f"Loaded {len(images)} images.")
print("First 5 labels:", labels[:5])

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)

X_train, X_val, y_train, y_val = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, Validation labels shape: {y_val.shape}")

model = Sequential([
    Conv2D(32, (3, 3), activation='tanh', input_shape=(128, 128, 4)),
    MaxPooling2D(pool_size=(2, 2)),
    # BatchNormalization(),

    Conv2D(64, (3, 3), activation='tanh'),
    MaxPooling2D(pool_size=(2, 2)),
    # BatchNormalization(),

    Conv2D(128, (3, 3), activation='tanh'),
    MaxPooling2D(pool_size=(2, 2)),
    # BatchNormalization(),

    Flatten(),
    # Dense(128, activation='relu'),
    # Dropout(0.5),

    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(18, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=32)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('my_model.keras')
