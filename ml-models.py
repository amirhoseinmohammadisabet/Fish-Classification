import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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

X_resized = [cv2.resize(img, (100, 100)) for img in X]
X_array = np.array(X_resized)
y_array = np.array(y)

# Flatten the images for input
n_samples, height, width, channels = X_array.shape
X_flat = X_array.reshape(n_samples, height * width * channels)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_array)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

# Normalize pixel values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel='linear', C=1, random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"--- {model_name} ---")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print('Classification Report:\n', class_report)

"""
--- Logistic Regression ---
Test Accuracy: 72.05%
Classification Report:
                precision    recall  f1-score   support

     bodianus       0.62      0.40      0.49        25
cephalopholis       0.57      0.54      0.55        24
 cirrhilabrus       0.94      0.80      0.86        20
        coris       0.86      0.46      0.60        26
  epinephelus       0.81      0.81      0.81        63
  halichoeres       0.55      0.86      0.67        36
    lethrinus       0.71      0.62      0.67        32
     lutjanus       0.72      0.77      0.74        62
 pseudanthias       0.78      0.91      0.84        34
   thalassoma       0.78      0.72      0.75        25

     accuracy                           0.72       347
    macro avg       0.73      0.69      0.70       347
 weighted avg       0.73      0.72      0.72       347

--- Naive Bayes ---
Test Accuracy: 33.72%
Classification Report:
                precision    recall  f1-score   support

     bodianus       0.41      0.28      0.33        25
cephalopholis       0.26      0.46      0.33        24
 cirrhilabrus       0.37      0.55      0.44        20
        coris       0.30      0.42      0.35        26
  epinephelus       0.41      0.33      0.37        63
  halichoeres       0.00      0.00      0.00        36
    lethrinus       0.31      0.53      0.39        32
     lutjanus       0.56      0.08      0.14        62
 pseudanthias       0.37      0.68      0.47        34
   thalassoma       0.29      0.44      0.35        25

     accuracy                           0.34       347
    macro avg       0.33      0.38      0.32       347
 weighted avg       0.35      0.34      0.30       347

--- Decision Tree ---
Test Accuracy: 34.01%
Classification Report:
                precision    recall  f1-score   support

     bodianus       0.30      0.24      0.27        25
cephalopholis       0.22      0.21      0.21        24
 cirrhilabrus       0.11      0.10      0.10        20
        coris       0.20      0.15      0.17        26
  epinephelus       0.51      0.43      0.47        63
  halichoeres       0.29      0.31      0.30        36
    lethrinus       0.28      0.31      0.29        32
     lutjanus       0.36      0.39      0.38        62
 pseudanthias       0.58      0.74      0.65        34
   thalassoma       0.14      0.16      0.15        25

     accuracy                           0.34       347
    macro avg       0.30      0.30      0.30       347
 weighted avg       0.34      0.34      0.34       347

--- Random Forest ---
Test Accuracy: 61.67%
Classification Report:
                precision    recall  f1-score   support

     bodianus       0.64      0.36      0.46        25
cephalopholis       0.52      0.46      0.49        24
 cirrhilabrus       1.00      0.45      0.62        20
        coris       1.00      0.15      0.27        26
  epinephelus       0.62      0.79      0.70        63
  halichoeres       0.43      0.67      0.52        36
    lethrinus       0.70      0.72      0.71        32
     lutjanus       0.55      0.73      0.63        62
 pseudanthias       0.79      0.88      0.83        34
   thalassoma       0.90      0.36      0.51        25

     accuracy                           0.62       347
    macro avg       0.72      0.56      0.57       347
 weighted avg       0.68      0.62      0.60       347

--- K-Nearest Neighbors ---
Test Accuracy: 56.48%
Classification Report:
                precision    recall  f1-score   support

     bodianus       0.70      0.28      0.40        25
cephalopholis       0.42      0.67      0.52        24
 cirrhilabrus       0.68      0.65      0.67        20
        coris       0.61      0.42      0.50        26
  epinephelus       0.73      0.56      0.63        63
  halichoeres       0.45      0.69      0.55        36
    lethrinus       0.40      0.59      0.48        32
     lutjanus       0.56      0.48      0.52        62
 pseudanthias       0.77      0.88      0.82        34
   thalassoma       0.53      0.40      0.45        25

     accuracy                           0.56       347
    macro avg       0.59      0.56      0.55       347
 weighted avg       0.59      0.56      0.56       347

--- Support Vector Machine ---
Test Accuracy: 73.20%
Classification Report:
                precision    recall  f1-score   support

     bodianus       0.82      0.56      0.67        25
cephalopholis       0.61      0.58      0.60        24
 cirrhilabrus       0.94      0.85      0.89        20
        coris       0.68      0.58      0.62        26
  epinephelus       0.74      0.79      0.76        63
  halichoeres       0.57      0.78      0.66        36
    lethrinus       0.68      0.78      0.72        32
     lutjanus       0.75      0.74      0.75        62
 pseudanthias       0.84      0.91      0.87        34
   thalassoma       0.93      0.56      0.70        25

     accuracy                           0.73       347
    macro avg       0.76      0.71      0.73       347
 weighted avg       0.75      0.73      0.73       347
 """