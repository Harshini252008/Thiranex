# DATA CLEANING & VISUALIZATION PROJECT

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests

# Load Dataset
file_name = "Titanic-Dataset.csv"

# Check if the file exists, if not, download it
if not os.path.exists(file_name):
    print(f"'{file_name}' not found. Attempting to download from a public source...")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"'{file_name}' downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download '{file_name}': {e}")
        print("Please ensure you have an active internet connection or upload the dataset manually.")
        raise FileNotFoundError(f"Unable to download {file_name}") from e

df = pd.read_csv(file_name)

# Display First 5 Rows
print("FIRST 5 ROWS")
print(df.head())

# Dataset Information
print("\nDATASET INFO")
print(df.info())

# Check Missing Values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Fill Missing Values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin Column (Too Many Missing Values)
df.drop('Cabin', axis=1, inplace=True)

# Remove Duplicate Rows
df.drop_duplicates(inplace=True)

# Check After Cleaning
print("\nAFTER CLEANING")
print(df.isnull().sum())

# Basic Statistics
print("\nSTATISTICS")
print(df.describe())

# -------------------------------
# DATA VISUALIZATION
# -------------------------------

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# 2. Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

# 3. Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# 4. Passenger Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 6. Survival Based on Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Based on Gender")
plt.show()

# Save Cleaned Dataset
df.to_csv("Cleaned_Titanic_Dataset.csv", index=False)

print("\nPROJECT COMPLETED SUCCESSFULLY")