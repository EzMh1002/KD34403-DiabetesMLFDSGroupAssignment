import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import arff
        
# Load dataset
data, meta = arff.loadarff(r"dataset_37_diabetes.arff")
df = pd.DataFrame(data)

# 2. INITIAL CLEANING 
# Decode byte strings from the ARFF format to standard strings
df['class'] = df['class'].str.decode('utf-8')

# 3. HANDLING MISSING VALUES (IMPUTATION)
# Medical attributes where 0 is a placeholder for missing data
# Columns: Glucose (plas), Blood Pressure (pres), Skin Thickness (skin), Insulin (insu), BMI (mass)
columns_zero = ['plas', 'pres', 'skin', 'insu', 'mass']

# Replace 0 with NaN so they don't skew the median calculation
df[columns_zero] = df[columns_zero].replace(0, np.nan)

# Put NaN values with the Median of each column 
for col in columns_zero:
    df[col] = df[col].fillna(df[col].median())

# 4. ENCODING & REFINING COLUMNS 
# Create 'target' column: tested_positive -> 1, tested_negative -> 0
df['target'] = df['class'].map({'tested_positive': 1, 'tested_negative': 0})

# Remove the redundant 'class' column as requested
df.drop(columns=['class'], inplace=True)

# 5. SHOW DATASET (Milestone 1 Requirement)
print("     Milestone 1: Data Pipeline Output")
print("------------------------------------------")
print("\nFirst 5 rows of the cleaned dataset:")
print(df.head())

print("\nDataset Summary Information:")
print("------------------------------")
print(df.info())

plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=['target']).corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('correlation_heatmap.png')

# 6. SAVE CLEANED DATA
df.to_csv('cleaned_diabetes_data.csv', index=False)
print("\nSuccess: 'cleaned_diabetes_data.csv' has been generated.")
