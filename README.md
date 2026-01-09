# 🚦 Traffic Sign Recognition System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-yellow?logo=python)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-blueviolet?logo=python)
![Pillow](https://img.shields.io/badge/Pillow-8.x-lightgrey?logo=python)
![Tqdm](https://img.shields.io/badge/Tqdm-4.x-brightgreen?logo=tqdm)


A deep learning–based end-to-end system designed to classify traffic signs using Convolutional Neural Networks (CNNs).  
The model is trained to recognize 43 different traffic sign classes and can predict signs from real-world images.

This project demonstrates the complete machine learning workflow, from data exploration to deployment.

---

## 📦 Dataset

The dataset used in this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**.

It is publicly available on Kaggle and contains **43 different traffic sign classes** collected from real-world driving scenarios.

🔗 Dataset link:  
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

This dataset includes:
- Labeled training images  
- Test images  
- Metadata files  
- Region of interest (ROI) information  

It is widely used for benchmarking traffic sign recognition systems.

---

## 📌 Project Overview

This project focuses on the automation of traffic sign identification, a crucial component in modern driver-assistance and autonomous driving systems.  
The system processes images of traffic signs and predicts their correct category using a trained deep learning model.

It includes:
- Exploratory Data Analysis (EDA)
- Image preprocessing and augmentation
- CNN model building and training
- Model evaluation and saving
- Deployment using Streamlit

---

## 📁 Project Structure

best_traffic_sign_model.keras   → Trained CNN model  
main2.ipynb                     → Complete notebook (EDA + Training + Evaluation)  
requirements.txt                → Required Python libraries  
traffic_app.py                  → Streamlit web application  
traffic_report.pdf              → Detailed project report  

---

## 🧠 Notebook Breakdown (main2.ipynb)

### 1. Data Inspection & Cleaning
Checks missing values, duplicates, data types, and overall integrity of the dataset.

### 2. Exploratory Data Analysis (EDA)
- Visualizes class distribution  
- Analyzes image dimensions  
- Builds correlation matrices  
- Studies dataset balance and structure  

### 3. Image Preprocessing
- Image reading using OpenCV  
- Resizing to a fixed input shape  
- Normalization of pixel values  
- One-hot encoding of labels  
- Train-test splitting  
- Data augmentation

### 4. CNN Model Architecture
The CNN includes:
- Convolutional layers  
- Batch normalization  
- Max pooling  
- Dropout layers  
- Fully connected dense layers  
- Softmax output layer (43 classes)

### 5. Model Training
- Compiled using Adam optimizer  
- Categorical cross-entropy loss  
- Accuracy metric  
- Trained on augmented image data  
- Validation tracking

### 6. Model Evaluation
Plots and metrics are used to analyze training and validation performance.

### 7. Model Saving
The trained model is saved as:
best_traffic_sign_model.keras

### 8. Prediction System
Implements a full prediction pipeline that:
- Loads an image  
- Preprocesses it  
- Predicts its class  
- Displays the traffic sign name  

---

## 🌐 Streamlit Web App (traffic_app.py)

The Streamlit application allows users to:
- Upload a traffic sign image  
- Automatically preprocess the image  
- Load the trained model  
- Get instant predictions  
- View the predicted sign label  

---

## ⭐ Key Features

- End-to-end deep learning pipeline  
- Strong exploratory data analysis  
- Custom CNN architecture  
- Data augmentation  
- 43 traffic sign classes  
- Saved trained model  
- Interactive web application  
- Real-world computer vision use case  

---

## ⚙️ Installation

pip install -r requirements.txt

---

## ▶️ Run Notebook
After downloading from the given link above and importing the datasets
Open main2.ipynb and run all cells in order.

---

## ▶️ Run Web App

streamlit run traffic_app.py (Given)

---

## 🎯 Outcome

This project demonstrates how deep learning can be applied to build a real-world traffic sign recognition system, covering everything from raw data to deployment.

---

## 👨‍💻 Author

Ghulam Murtaza 24L-2566 
Yousaf Iqbal 24L-2539 
Hannan Khan 24L-2550 
FAST NUCES Lahore – Data Science
