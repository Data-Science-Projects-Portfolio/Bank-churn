import pandas as pd

# Load dataset
df = pd.read_csv("Bank Customer Churn Prediction.csv")

# Display first records
print(df.head())

# Dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Column information
print("\nColumn Information:")
print(df.info())
