# Econometric Model for EdTech Policy Analysis
# Purpose: Predict literacy rates based on internet access, device ownership, and teacher training
# Prepared for Pareto & IISER Bhopal Economics Club Case Study Competition
# Date: April 13, 2025

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Simulate a dataset (since real data from ASER/UDISE+ is not directly accessible)
# Variables: Literacy Rate (%), Internet Access (%), Device Ownership (%), Teacher Training (hours/year)
n = 500  # Number of observations (e.g., districts)
data = {
    'literacy_rate': np.random.normal(75, 10, n),  # Mean literacy rate ~75%
    'internet_access': np.random.uniform(20, 80, n),  # % of schools with internet
    'device_ownership': np.random.uniform(10, 70, n),  # % of students with devices
    'teacher_training': np.random.uniform(10, 100, n)  # Training hours
}

# Introduce some realistic correlations
data['literacy_rate'] = (
    0.4 * data['internet_access'] +
    0.3 * data['device_ownership'] +
    0.2 * data['teacher_training'] +
    np.random.normal(0, 5, n)
)

# Create DataFrame
df = pd.DataFrame(data)

# Step 2: Exploratory Data Analysis
print("Dataset Preview:")
print(df.head())
print("\nCorrelation Matrix:")
print(df.corr())

# Visualize correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Variables')
plt.savefig('correlation_matrix.png')
plt.close()

# Step 3: Prepare data for regression
X = df[['internet_access', 'device_ownership', 'teacher_training']]
y = df['literacy_rate']

# Add constant for intercept
X = sm.add_constant(X)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Run OLS regression
model = sm.OLS(y_train, X_train).fit()

# Print regression summary
print("\nRegression Summary:")
print(model.summary())

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\nR-squared on Test Set: {r2:.3f}")

# Step 6: Visualize regression results
plt.figure(figsize=(10, 6))
for var in ['internet_access', 'device_ownership', 'teacher_training']:
    plt.scatter(X_test[var], y_test, label='Actual', alpha=0.5)
    plt.scatter(X_test[var], y_pred, label='Predicted', alpha=0.5)
    plt.xlabel(var)
    plt.ylabel('Literacy Rate (%)')
    plt.legend()
    plt.title(f'Actual vs Predicted Literacy Rate by {var}')
    plt.savefig(f'scatter_{var}.png')
    plt.close()

# Step 7: Interpret key findings
print("\nKey Findings:")
print(f"A 10% increase in internet access is associated with a {model.params['internet_access']*10:.2f}% increase in literacy rate.")
print(f"Model explains {model.rsquared*100:.1f}% of variance in literacy rates.")

# Save dataset for reproducibility
df.to_csv('edtech_data.csv', index=False)
print("\nDataset saved as 'edtech_data.csv'.")
