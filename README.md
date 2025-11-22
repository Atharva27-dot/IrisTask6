# KNN Classification - Iris Dataset

This repository contains a complete implementation of the **K-Nearest Neighbors (KNN)** algorithm applied on the **Iris dataset**, along with accuracy evaluation, confusion matrix, decision boundary visualization, and saved text results.

---

## 📌 Features Implemented

### ✔ Load & preprocess dataset  
- Iris dataset from sklearn  
- Train/test split  
- Feature scaling using StandardScaler  

### ✔ KNN Model  
- Trains multiple K values (1–15)  
- Selects best K based on accuracy  
- Final evaluation & predictions  

### ✔ Visualization  
- Accuracy vs K plot  
- Confusion matrix heatmap  
- Decision boundary visualization (using 2 features)  

### ✔ Saved Results  
All text-based outputs are stored in:

```
results.txt
```

This includes:
- Accuracy for each K  
- Best K  
- Confusion matrix  
- Classification report  

---

## 📂 Project Files

```
│
├── knn_iris_classification.py   # Main program (KNN + visuals + saving results)
├── results.txt                  # Auto-generated text results
└── README.md                    # Project documentation
```

---

## ▶️ How to Run the Project

### **1. Install required libraries**

```sh
pip install numpy matplotlib scikit-learn
```

### **2. Run the script**

```sh
python knn_iris_classification.py
```

### **3. View Results**
- Text results → `results.txt`
- Plots → open automatically  
- Code output → terminal  

---

## 🚀 GitHub Upload Commands

If you want to push this project to GitHub, use:

```sh
git init
git add .
git commit -m "Initial commit - KNN Iris Classification"
git branch -M main
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main
```

---

## 📧 Author
Atharva Jadhav  
KNN Classification - Machine Learning Internship Task


