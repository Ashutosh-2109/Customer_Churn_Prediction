# ==================================================
# Customer Churn Prediction (EDA + ML + GUI)
# CodeChef Club Entry Project
# Author: Ashutosh Pandey
# Model Version: v1.0
# ==================================================

# ---------- IMPORT LIBRARIES ----------
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tkinter as tk
from tkinter import messagebox, ttk

sns.set(style="whitegrid")


# ==================================================
# MODEL VERSIONING
# ==================================================
MODEL_VERSION = "v1.0"


# ==================================================
# LOAD DATA
# ==================================================
df = pd.read_csv(r"C:\Users\91932\OneDrive\Documents\Engineering\CODING\Projects\New folder\Customer-Churn.csv")
print("Dataset Shape:", df.shape)


# ==================================================
# EDA
# ==================================================
plt.figure(figsize=(5,4))
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Churn vs Tenure")
plt.show()


# ==================================================
# DATA PREPROCESSING
# ==================================================
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

df.drop('customerID', axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)


# ==================================================
# TRAIN-TEST SPLIT
# ==================================================
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==================================================
# SCALING
# ==================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==================================================
# MODEL TRAINING
# ==================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    random_state=42
)
rf.fit(X_train, y_train)


# ==================================================
# MODEL EVALUATION
# ==================================================
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nModel Performance:")
print("Accuracy :", accuracy)
print("Recall   :", recall)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


# ==================================================
# GUI PREDICTION FUNCTION
# ==================================================
def predict_churn():
    try:
        tenure = float(entry_tenure.get())
        monthly = float(entry_monthly.get())
        total = float(entry_total.get())
        contract = contract_var.get()

        one_year = 1 if contract == "One year" else 0
        two_year = 1 if contract == "Two year" else 0

        input_df = pd.DataFrame([[
            tenure, monthly, total, one_year, two_year
        ]], columns=[
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract_One year', 'Contract_Two year'
        ])

        for col in X.columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[X.columns]
        input_scaled = scaler.transform(input_df)

        prob = rf.predict_proba(input_scaled)[0][1]
        pred = 1 if prob >= 0.5 else 0

        # Update result text
        if pred == 1:
            result_label.config(text="⚠ Customer WILL Churn", fg="red")
        else:
            result_label.config(text="✅ Customer will NOT Churn", fg="green")

        # Update probability bar
        prob_bar['value'] = prob * 100
        prob_label.config(text=f"Churn Probability: {prob:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# ==================================================
# TKINTER GUI
# ==================================================
root = tk.Tk()
root.title("Customer Churn Predictor")
root.geometry("450x550")
root.configure(bg="#f2f2f2")

tk.Label(
    root,
    text=f"Customer Churn Prediction\nModel: Random Forest | {MODEL_VERSION}\nAccuracy: {accuracy:.2f} | Recall: {recall:.2f}",
    font=("Arial", 12, "bold"),
    bg="#f2f2f2"
).pack(pady=10)

tk.Label(root, text="Tenure (months)", bg="#f2f2f2").pack()
entry_tenure = tk.Entry(root)
entry_tenure.pack()

tk.Label(root, text="Monthly Charges", bg="#f2f2f2").pack()
entry_monthly = tk.Entry(root)
entry_monthly.pack()

tk.Label(root, text="Total Charges", bg="#f2f2f2").pack()
entry_total = tk.Entry(root)
entry_total.pack()

tk.Label(root, text="Contract Type", bg="#f2f2f2").pack()
contract_var = tk.StringVar(value="Month-to-month")

tk.OptionMenu(root, contract_var,
              "Month-to-month", "One year", "Two year").pack()

tk.Button(root, text="Predict Churn",
          command=predict_churn,
          bg="#4CAF50", fg="white",
          font=("Arial", 12)).pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f2f2f2")
result_label.pack(pady=5)

# Probability bar
tk.Label(root, text="Prediction Confidence", bg="#f2f2f2").pack()
prob_bar = ttk.Progressbar(root, length=300, mode='determinate')
prob_bar.pack(pady=5)

prob_label = tk.Label(root, text="", bg="#f2f2f2")
prob_label.pack()

root.mainloop()
