# Every Team Member Should Follow This Process:
# Update main branch
# git checkout main
# git pull origin main

# Create task branch
#git checkout -b feature/my-task

# Do work

# Save changes
# git add .

# Commit
# git commit -m "Completed task"

# Push
# git push origin feature/my-task

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Bank Customer Churn Prediction.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nChurn Distribution:")
print(df["churn"].value_counts())

sns.countplot(x="churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()



