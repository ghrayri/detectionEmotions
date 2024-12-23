#importation des biblio
import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report

# Definir les émotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
num_classes = len(emotions)


# fonctions pour load dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for emotion in emotions:
        emotion_folder = os.path.join(folder, emotion)
        for filename in os.listdir(emotion_folder):
            img = cv2.imread(os.path.join(emotion_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(emotions.index(emotion))
    return np.array(images), np.array(labels)
   

# load trainer et tester 
train_images, train_labels = load_images_from_folder("C:/Users/user/.cache/kagglehub/datasets/msambare/fer2013/versions/1/train")
test_images, test_labels = load_images_from_folder("C:/Users/user/.cache/kagglehub/datasets/msambare/fer2013/versions/1/test")

# Normaliser  images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# Convertir  labels to categorical
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# generation data
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(train_images)

# Build model
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compilation
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Trainer model
model.fit(datagen.flow(train_images, train_labels, batch_size=64),
          epochs=50,
          validation_data=(test_images, test_labels),
          callbacks=[reduce_lr, early_stopping])

# Sauvegarder
model.save('emotion_model3.h5')

# Evaluation du performance
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# afficher rapport du  classification
print(classification_report(true_classes, predicted_classes, target_names=emotions))
