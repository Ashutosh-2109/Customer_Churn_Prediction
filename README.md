# Customer Churn Prediction using Machine Learning

## 📖 Overview
Customer churn refers to customers who stop using a company’s service.  
This project builds an **end-to-end machine learning system** to predict customer churn using demographic, service, and billing information.

The project covers the complete ML pipeline:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training and evaluation
- Handling class imbalance
- Model interpretation
- Optional inference for new customers

This project was developed as part of a **CodeChef Club recruitment task**.

---

## 🎯 Objective
To predict whether a customer will churn using supervised machine learning models and to analyze the key factors influencing churn.

---

## 📊 Dataset
- **Dataset:** Customer Churn Dataset (CSV)
- **Target Variable:** `Churn` (Yes / No)
- **Features include:**
  - Customer tenure
  - Contract type
  - Monthly and total charges
  - Services subscribed
  - Payment methods

---

## 🔍 Exploratory Data Analysis (EDA)
EDA was performed to understand:
- Churn distribution and class imbalance
- Relationship between churn and tenure
- Effect of contract type on churn
- Billing behavior of churned vs non-churned customers

Key observations:
- Customers with **shorter tenure** churn more
- **Month-to-month contracts** have higher churn
- Higher **monthly charges** increase churn risk

---

## 🧹 Data Preprocessing
- Converted `TotalCharges` to numeric and handled missing values
- Removed irrelevant identifiers (`customerID`)
- One-hot encoded categorical variables
- Performed train–test split with stratification
- Scaled numerical features using `StandardScaler`

---

## 🤖 Models Used

### 1️⃣ Random Forest Classifier (Baseline)
- Robust on tabular data
- Provides feature importance for interpretability

### 2️⃣ XGBoost + SMOTE (Improved Model)
- **SMOTE** used to handle class imbalance
- **XGBoost** used for better performance on structured data
- Focused on improving **recall** for churned customers

---

## 📈 Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Special emphasis was placed on **recall**, as missing a churned customer is costly in real-world scenarios.

---

## 🔑 Feature Importance
Feature importance analysis showed that:
- **Tenure**
- **Contract type**
- **Monthly charges**

are the most influential factors in predicting customer churn.

---

## ⚠️ Limitations
- Dataset is moderately imbalanced
- Temporal customer behavior is not captured
- Some features may be correlated

---

## 🚀 Possible Improvements
- Use advanced resampling techniques or cost-sensitive learning
- Try LightGBM or CatBoost
- Perform threshold tuning instead of fixed 0.5 cutoff
- Deploy as a web application using Flask or FastAPI

---

## 🧪 Optional Inference
An optional section allows users to test the trained model on **new customer data** and view:
- Churn prediction
- Churn probability

This demonstrates real-world applicability beyond model training.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Google Colab

---

## 📂 Repository Structure
