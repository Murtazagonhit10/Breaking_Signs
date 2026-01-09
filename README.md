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



A Deep Learning-based end-to-end system designed to classify traffic signs with high precision. This project utilizes a **Convolutional Neural Network (CNN)** trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset and features a real-time web interface built with **Streamlit**.

---

## 📸 Project Overview

This project focuses on the automation of traffic sign identification, a crucial component for Autonomous Vehicles and Advanced Driver Assistance Systems (ADAS). It handles the entire pipeline: from data preprocessing and CNN architecture design to model evaluation and deployment.

### ✅ Key Features
- **43 Different Classes:** Recognizes everything from speed limits and stop signs to danger warnings.
- **Large Scale Training:** Trained on over **40,000 images** from the GTSRB dataset.
- **Optimized CNN Architecture:** Uses multiple Convolutional layers, Max-Pooling, and Dropout for high accuracy and minimal overfitting.
- **Real-Time Deployment:** Includes an interactive **Streamlit Web App** where users can upload images for instant classification.
- **Performance Visualization:** Generates accuracy and loss graphs to monitor model convergence.

---

## 🕹️ App Controls & Usage

| Component | Function |
|-----------|----------|
| **Image Uploader** | Drag and drop or browse local JPG/PNG files of traffic signs. |
| **Predict Button** | Triggers the CNN model to analyze the uploaded image. |
| **Classification Result** | Displays the predicted name of the traffic sign. |
| **Confidence Score** | Shows a percentage indicating how "sure" the model is of its prediction. |

---

## 🛠️ Technologies Used

- **Deep Learning Framework:** TensorFlow & Keras
- **Image Processing:** OpenCV & PIL (Pillow)
- **Data Manipulation:** Pandas & NumPy
- **Machine Learning Utilities:** Scikit-learn (Train-Test Split)
- **Visualization:** Matplotlib & Seaborn
- **Deployment:** Streamlit

---

## 🏗️ Model Architecture

The model is built using a sequential CNN approach:
1. **Convolutional Layers:** Extracts spatial features (edges, shapes) from images.
2. **Max-Pooling:** Reduces dimensionality and computational load.
3. **Dropout Layers:** Prevents overfitting by randomly deactivating neurons during training.
4. **Flatten & Dense Layers:** Fully connected layers that perform the final classification into 43 categories.
5. **Activation (Softmax):** Outputs a probability distribution across all classes.

---

## ▶️ How to Run

### 1. Prerequisites
Ensure you have Python installed. Install the necessary libraries using:
```bash
pip install tensorflow streamlit opencv-python pandas scikit-learn matplotlib tqdm
