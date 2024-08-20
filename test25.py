import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from transformers import ViTFeatureExtractor, TFViTForImageClassification
import matplotlib.pyplot as plt
import seaborn as sns

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
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)
datagen.fit(X_train)

# Load the ViT model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
base_model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(top10dict))

# Preprocess input for the ViT model
def preprocess_images(images):
    images = images.astype(np.float32)
    inputs = feature_extractor(images, return_tensors="np")
    return inputs['pixel_values']

# Preprocess training and test data
X_train_preprocessed = preprocess_images(X_train)
X_test_preprocessed = preprocess_images(X_test)

# Call the model on a batch of data to build it
_ = base_model(X_train_preprocessed)

# Print the model summary
base_model.summary()

# Compile the model
base_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for learning rate adjustment and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = base_model.fit(datagen.flow(X_train_preprocessed, y_train, batch_size=32),
                    epochs=50,
                    validation_data=(X_test_preprocessed, y_test),
                    callbacks=[reduce_lr, early_stopping])

# Evaluate the model
test_loss, test_accuracy = base_model.evaluate(X_test_preprocessed, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict the labels for the test set
y_pred = base_model.predict(X_test_preprocessed)
y_pred_classes = np.argmax(y_pred.logits, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot training & validation accuracy values
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot confusion matrix
plt.subplot(1, 3, 3)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()
