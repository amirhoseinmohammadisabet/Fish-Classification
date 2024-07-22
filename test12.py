import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

# Directory Path    
directory_path = "Data/Fish_Data/images/cropped/"
cnt = {}
for filename in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, filename)):
        name = filename.split("_")[0]
        cnt[name] = cnt.get(name, 0) + 1
print(f"Number of labels: {len(cnt)}")

sorted_dict = dict(sorted(cnt.items(), key=lambda item: item[1], reverse=True))
top10dict = dict(list(sorted_dict.items())[:10])

# Load Images and Labels
X = []
y = []
for filename in os.listdir(directory_path):
    if os.path.isfile(os.path.join(directory_path, filename)):
        name = filename.split("_")[0]
        if name in top10dict:
            X.append(cv2.imread(os.path.join(directory_path, filename)))
            y.append(name)

# Resize images to a common size (224x224)
X_resized = [cv2.resize(img, (224, 224)) for img in X]
X_array = np.array(X_resized)
y_array = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_array)
y_categorical = to_categorical(y_encoded)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_array, y_categorical, test_size=0.2, random_state=42)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.3,
    shear_range=0.3,
    brightness_range=[0.8,1.2],
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build the CNN model with transfer learning (EfficientNetB0)
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(top10dict), activation='softmax')
])

# Unfreeze the top layers of the base model for fine-tuning
for layer in base_model.layers[-10:]:
    layer.trainable = True

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Callbacks for learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)

# Train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=50,
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, early_stopping, lr_scheduler])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
