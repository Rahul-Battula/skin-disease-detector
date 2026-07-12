import pandas as pd

# Load metadata
df = pd.read_csv('../data/raw/HAM10000_metadata.csv')

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# Class distribution
print("\nDisease class distribution:")
print(df['dx'].value_counts())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())