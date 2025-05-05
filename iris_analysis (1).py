# Iris Dataset Analysis
# Converted from notebook to script format

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map(dict(zip(range(3), iris.target_names)))
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

print(df.head())
print(df.dtypes)
print(df.isnull().sum())
df = df.dropna()
print(df.describe())
print(df.groupby('species').mean())
sns.set(style='whitegrid')
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title('Trend of Petal Length Over Index')
plt.xlabel('Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
plt.hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()
