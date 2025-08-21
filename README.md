# 🥭🍅 Fruits and Vegetables Image Classification

A simple deep learning project that classifies images of fruits and vegetables using **Convolutional Neural Networks (CNNs)** with TensorFlow & Keras.  
This is one of my first machine learning projects to build my GitHub portfolio.

---

## 📌 Project Overview
The goal of this project is to train a neural network model that can distinguish between different categories of images.  
Currently, I used the **TensorFlow Flower Dataset** (for demo), but this project can be extended to any fruits and vegetables dataset.

---

## 🚀 Features
- Loads image dataset from local folder
- Preprocesses images (resize, normalization)
- Builds a **CNN model**
- Trains and evaluates on training/validation sets
- Plots accuracy graph
- Saves the trained model as `.h5` file

---

## 🛠️ Tech Stack
- Python 3
- TensorFlow / Keras
- Matplotlib
- Git & GitHub

---

## 📂 Project Structure
```
fruits-vegetables-classification/
│
├── fruits_vegetables_classification.py   # Main code file
├── fruits_vegetables_model.h5            # Saved trained model (after running code)
├── requirements.txt                      # Dependencies
└── README.md                             # Project documentation
```

---

## ⚙️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fruits-vegetables-classification.git
   cd fruits-vegetables-classification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python fruits_vegetables_classification.py
   ```

4. The model will train and save as:
   ```
   fruits_vegetables_model.h5
   ```

---

## 📊 Sample Training Result
The model shows increasing accuracy over epochs:

![accuracy-plot](https://user-images.githubusercontent.com/000000/0000000/sample-accuracy.png)

---

## 🔮 Future Improvements
- Replace dataset with **fruits & vegetables dataset**  
- Add **prediction script** for single image testing  
- Deploy model as a **web app** with Flask/Streamlit  

---

## ✨ Author
**Ahmad** – Learning Machine Learning & building projects to strengthen my GitHub profile.  
