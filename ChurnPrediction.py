# ==============================================
# EVERY TEAM MEMBER SHOULD FOLLOW THIS PROCESS EVERYTIME THEY WORK ON THE PROJECT 
# ==============================================

# UPDATE MAIN BRANCH
# ---------------------
# git checkout main
# git pull origin main

# CREATE TASK BRANCH
# ---------------------
# git checkout -b feature/my-task

# DO WORK
# ---------------------

# SAVE CHANGES
# ---------------------
# git add .

# COMMIT CHANGES
# ---------------------
# git commit -m "Completed task"

# PUSH CHANGES
# ---------------------
# git push origin feature/my-task

# =============================================================================
# PHASE 1: BUSINESS UNDERSTANDING
# =============================================================================
#
# PURPOSE
# -------
# The objective of this project is to develop a machine learning model that
# predicts customer churn. Churn refers to customers who discontinue their
# relationship with the company (e.g., cancel a subscription, stop purchasing,
# or switch to a competitor).
#
# WHY THIS MATTERS
# ----------------
# Retaining existing customers is typically less expensive than acquiring new
# ones. By identifying customers who are likely to leave before they churn,
# the business can proactively intervene with retention strategies such as:
#
# - Personalized offers
# - Loyalty incentives
# - Customer service engagement
# - Contract reviews
# - Pricing adjustments
#
# BUSINESS QUESTIONS TO ANSWER
# ----------------------------
#
# 1. Which customers are at risk of churn?
#    - Identify high-risk customers before they leave.
#
# 2. What factors drive churn?
#    - Determine the key variables influencing customer decisions.
#    - Examples: tenure, contract type, monthly charges, support calls,
#      payment method, product usage, etc.
#
# 3. When should intervention occur?
#    - Identify the optimal point in the customer lifecycle where retention
#      efforts can have maximum impact.
#
# 4. Which customers should retention teams prioritize?
#    - Rank customers by churn probability so limited resources can be focused
#      on the customers most likely to leave.
#
# TECHNICAL SUCCESS CRITERIA
# --------------------------
#
# ROC-AUC > 0.80
# - Indicates strong ability to distinguish between churners and non-churners.
#
# Recall > 80% for Churn Class
# - The model should correctly identify at least 80% of customers who
#   eventually churn.
# - Recall is particularly important because missing a churner can directly
#   result in lost revenue.
#
# BUSINESS SUCCESS CRITERIA
# -------------------------
#
# 1. Reduce overall customer churn rate.
#
# 2. Improve customer retention performance across targeted customer segments.
#
# 3. Increase Customer Lifetime Value (CLV) by retaining customers longer.
#
# EXPECTED OUTPUT
# ---------------
# The final solution should produce:
#
# - Churn probability scores for each customer.
# - Risk categories (Low, Medium, High).
# - Key churn drivers and explanations.
# - Actionable insights for retention teams.
#
# DELIVERABLES
# ------------
#
# 1. Exploratory Data Analysis (EDA)
# 2. Feature Engineering Pipeline
# 3. Machine Learning Models
# 4. Model Evaluation Report
# 5. Explainability Analysis (SHAP)
# 6. Customer Risk Scoring Framework
# 7. Production-Ready Churn Prediction Model
#
# =============================================================================

# ====================================================================



# Purpose:
#--------------

# Predict customer churn using machine learning.

# ====================================================================

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
# Load dataset
df = pd.read_csv("Bank Customer Churn Prediction.csv")

# %%
#INSPECT THE DATASET
#==============================

# Data types
df.info()

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

# %%
# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# %%
# Missing values
df.isnull().sum().sort_values(ascending=False)
print("\nChurn Distribution:")
print(df["churn"].value_counts())

# %%
#Exploratory Data Analysis (EDA)
#==============================

sns.countplot(x="churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()
#Important questions:
#Is churn balanced?
#Is churn rare?

# %%
#NUMERICAL FEATURES
numerical_cols = df.select_dtypes(
    include=['int64','float64']
).columns

df[numerical_cols].hist(
    figsize=(15,10)
)

plt.tight_layout()
plt.show()

# %%
#CORRELATIION ANALYSIS
plt.figure(figsize=(12,8))

sns.heatmap(
    df[numerical_cols].corr(),
    cmap='coolwarm',
    annot=False
)

plt.title("Feature Correlation Matrix")
plt.show()

#Phase 3: Data Preparation
#=============================
#This phase typically consumes 70-80% of project time

# %%
# #Separate Features and Target
TARGET = "Churn"

X = df.drop(TARGET, axis=1)

y = df[TARGET]

# %%
#Encode Target
y = y.map({
    "Yes":1,
    "No":0
})

# %%
# #Identify Column Types
cat_cols = X.select_dtypes(
    include=['object']
).columns

num_cols = X.select_dtypes(
    exclude=['object']
).columns

print(cat_cols)
print(num_cols)

# %%
#Train Test Split
#Never look at test data until final evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Build Preprocessing Pipeline
# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            num_cols
        ),
        (
            "cat",
            categorical_pipeline,
            cat_cols
        )
    ]
)

#Phase 4: Modeling
#==============================
#We will intentionally build multiple models.
#Baseline Model
#==========================
#Logistic Regression
# %%
from sklearn.linear_model import LogisticRegression
baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

baseline_model.fit(
    X_train,
    y_train
)


#I have stopped here. I will continue, allow me to follow this setup and continue building the model. then we will discuss
#You can run the code, correct errors and see the results. Once you are done, we can discuss the next steps.

#Veronica's edit

#The next logical move is to test it on unseen data and measure whether it actually works.
# Howeever, before training/evaluation, confirm whether our CSV uses Churn or churn as the target variable. If it is churn, we need to change the TARGET variable to "churn" instead of "Churn".
#Our code mixes both: earlier: df["churn".] later: TARGET = "Churn". That mismatch can cause errors or wrong results. Fix that before moving on.

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = baseline_model.predict(X_test)
y_proba = baseline_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC:")
print(roc_auc_score(y_test, y_proba))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#this tells us:

#how many churners were correctly found
#how many were missed
#whether the model is actually useful
#Compare against our success criteria
#we wrote:

#ROC-AUC > 0.80
#Recall > 80% for churn class
#So after evaluation, check whether our model meets those thresholds.

#If it does not, then the next step is:

#tune hyperparameters
#try other models
#check class imbalance
#adjust evaluation strategy