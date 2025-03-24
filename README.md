# Comparative Analysis of ML/DL Models for Image Classification

## 📌 Project Overview  
This project evaluates and compares the performance of **traditional Machine Learning (ML) models** and **state-of-the-art Deep Learning (DL) architectures** on an image classification task. The goal was to benchmark accuracy and identify the best approach for the given dataset.

## 🔍 Key Findings  
- **Top-performing model**: **DenseNet121** (90.20% accuracy)  
- **Deep Learning > Traditional ML**: Pre-trained DL models (VGG16, Xception, MobileNetV2) outperformed ML models (SVM, Random Forest, etc.) by a wide margin.  
- **Custom CNN** achieved **80.00% accuracy**, demonstrating the value of tailored architectures.  

## 🛠️ Implemented Models  
### **Deep Learning**  
- Custom CNN (80.00%)  
- MobileNetV2 (83.57%)  
- VGG16 (87.61%)  
- DenseNet121 (90.20%)  
- Xception (87.61%)  
- ResNet50 (82.54%)  

### **Traditional ML**  
- SVM (73.20%)  
- Logistic Regression (72.05%)  
- Random Forest (61.67%)  
- KNN (56.48%)  
- Naive Bayes (33.72%)  

## 📊 Dataset & Preprocessing  
- **Top 10 labels** selected for balanced classification.  
- Images resized to **224x224**, normalized to [0, 1].  
- **Augmentation**: Rotation, flipping, zooming, shearing.  

## 🚀 Future Work  
- Hyperparameter tuning for DL models.  
- Testing Vision Transformers (ViTs).  
- Ensemble methods to boost accuracy.  

## 📂 Files  
- `models/`: Code for all implemented models.  
- `data_preprocessing.py`: Dataset cleaning and augmentation.  
- `results/`: Confusion matrices, accuracy/loss plots.  

---

🔗 **Full report available in [Reports/Final hopefully Project Report Comparative Analysis of Machine Learning and Deep Learning Models.pdf]()**
📫 **Contact**: [amirhoseinmohammadisabet@gmail.com](mailto:amirhoseinmohammadisabet@gmail.com)  
