# 🏥 Health Insurance Cross Sell Prediction

---

### 🔎 Overview

This repository contains an **end-to-end Machine Learning project** designed to predict **which existing Health Insurance customers are likely to be interested in Vehicle Insurance**.  
The project follows a **production-ready ML pipeline architecture**, including:

- Modularized components  
- Data preprocessing & transformation  
- Model training & tuning  
- Logging & exception handling  
- A complete prediction pipeline  
- A **Flask web application** for real-time inference  

This project was developed by following structured, industry-style ML tutorial practices and adapting them into a clean, extensible architecture.

---

### 📂 Dataset

- **Source:** https://www.kaggle.com/competitions/massp-health-insurance-prediction/data  
- **Description:**  
  The dataset contains customer demographics, vehicle details, and policy attributes.  
  The target is whether a customer is interested in purchasing **Vehicle Insurance**.

  Features include:
  - Gender, Age, Region  
  - Vehicle Damage History  
  - Vehicle Age  
  - Policy Premium  
  - Policy Sales Channel  
  - Previously Insured  
  - etc.

---

### ⚙️ Pipeline Stages

This project follows a structured ML pipeline:

#### **1️⃣ Data Ingestion**
- Reads raw data  
- Splits into train/test sets  
- Stores files inside `/artifacts`

#### **2️⃣ Data Transformation**
- Handles missing values  
- Encodes categorical features  
- Scales numerical features  
- Saves the preprocessing object  

#### **3️⃣ Model Training**
- Trains multiple ML models  
- Hyperparameter tuning using **Optuna**  
- Evaluates performance using ROC_AUC score  
- Saves the best model to artifacts  

#### **4️⃣ Prediction Pipeline**
- Loads saved preprocessor + model  
- Generates predictions for both:
  - Flask app input  
  - Batch predictions via script  

#### **5️⃣ Flask Application**
A simple web interface allowing users to input features and get instant predictions.

---

### 📊 Tools & Libraries

- **pandas, numpy** → Data handling  
- **matplotlib, seaborn** → Data visualization  
- **scikit-learn** → Preprocessing, modeling & evaluation  
- **Optuna** → Hyperparameter tuning  
- **pickle** → Saving/loading model artifacts  
- **Flask** → Web deployment  

---

### 📌 Project Structure

Healthcare-Insurance-Prediction/
│
├── notebooks/ # Jupyter notebooks for exploration & experiments
│ ├── data/ # Raw dataset(s)
│ └── *.ipynb
│
├── src/insurance/ # Main package for pipelines & components
│ ├── components/ # Data ingestion, transformation, model trainer
│ ├── pipelines/ # train_pipeline.py and predict_pipeline.py
│ ├── logger.py # Custom logging
│ └── exception.py # Centralized exception handling
│
├── artifacts/ # Stored model, scaler, preprocessed objects etc.
│
├── logs/ # Training & prediction logs
│
├── app.py # Flask application for deployed model
├── requirements.txt # Project dependencies
├── setup.py # Package definition
├── .gitignore
└── README.md

---