
# Lightweight CNN for Fashion-MNIST 👗👟👜

## 📌 Overview
This project implements a **lightweight Convolutional Neural Network (CNN)** for classifying Fashion-MNIST images into 10 clothing categories.  
The implementation is optimized for **memory efficiency** using:
- Chunk-based CSV loading
- Sampling to avoid memory overflow
- Simplified data augmentation
- Compact CNN architecture with dropout regularization

---

## 📊 Dataset
Dataset: [Fashion-MNIST (Kaggle)](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

- **Train set**: 60,000 grayscale images (28x28 pixels)  
- **Test set**: 10,000 grayscale images  
- **Classes**:
  1. T-shirt/top
  2. Trouser
  3. Pullover
  4. Dress
  5. Coat
  6. Sandal
  7. Shirt
  8. Sneaker
  9. Bag
  10. Ankle boot  


## ⚙️ Project Workflow
1. **Memory-Efficient Data Loading**
   - Load data in chunks (5,000 rows at a time).
   - Option to use a fraction of the dataset for testing on low-RAM systems.
2. **Preprocessing**
   - Normalize pixel values (0–1).
   - Simple augmentation: horizontal flip + validation split.
3. **Model Architecture**
   - Conv2D (32 filters, ReLU) → MaxPooling  
   - Conv2D (64 filters, ReLU) → MaxPooling  
   - Flatten → Dense(64, ReLU) → Dropout(0.3)  
   - Dense(10, Softmax)  
4. **Training**
   - Optimizer: Adam (lr=0.001)  
   - Loss: Sparse categorical crossentropy  
   - Callbacks: EarlyStopping, ReduceLROnPlateau  
5. **Evaluation**
   - Accuracy & Loss curves
   - Confusion matrix heatmap
   - Sample prediction visualization

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI-ML-Projects.git
   cd AI-ML-Projects/FashionMNIST-LightCNN
