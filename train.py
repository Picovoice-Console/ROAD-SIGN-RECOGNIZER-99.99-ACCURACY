# train.py
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

def load_data():
    data = []
    labels = []
    for i in range(43):
        folder_path = os.path.join('dataset/train', str(i))
        print(f"Loading folder: {folder_path}")  # Debug print
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if not os.path.isfile(img_path):
                continue  # Skip if not a file
            try:
                image = Image.open(img_path).convert('RGB').resize((30, 30))
                data.append(np.array(image))
                labels.append(i)
            except Exception as e:
                print(f"Error loading image: {img_path} | {e}")
    print(f"Loaded {len(data)} images.")  # Summary
    return np.array(data), np.array(labels)

def build_model():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(30, 30, 3)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(43, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_and_save():
    data, labels = load_data()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, 43)
    y_test = to_categorical(y_test, 43)

    model = build_model()
    history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

    model.save('traffic_sign_model.h5')
    print("Model trained and saved as traffic_sign_model.h5")

    # Plotting graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_and_save()
