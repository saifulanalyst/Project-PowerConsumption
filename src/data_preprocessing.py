from scipy.stats import zscore
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Summary statistics
print(df.describe())

# Correlation matrix to understand feature relationships
# Exclude non-numeric columns (e.g., 'Datetime')
corr = df.drop(columns=['Datetime']).corr()

# Plot the correlation matrix
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
# Check for missing values
print(df.isnull().sum())

# Handle missing values using interpolation (suitable for time series)
df = df.interpolate(method='linear')  # Linear interpolation
df = df.ffill().bfill()  # Forward-fill and backward-fill

# Check for inconsistent data (e.g., outliers)
print(df.describe())

# Remove outliers (e.g., values outside 3 standard deviations)

z_scores = zscore(df.select_dtypes(include=['float64', 'int64']))
# Keep rows where all z-scores are less than 3
df = df[(z_scores < 3).all(axis=1)]

# Verify cleaned data
print(df.isnull().sum())
print(df.describe())
# Scatter plots to analyze relationships
plt.figure(figsize=(18, 6))

# Temperature vs Power Consumption
plt.subplot(1, 3, 1)
sns.scatterplot(x=df['Temperature'], y=df['PowerConsumption_Zone1'])
plt.title("Temperature vs Power Consumption (Zone 1)")

# Humidity vs Power Consumption
plt.subplot(1, 3, 2)
sns.scatterplot(x=df['Humidity'], y=df['PowerConsumption_Zone1'])
plt.title("Humidity vs Power Consumption (Zone 1)")

# WindSpeed vs Power Consumption
plt.subplot(1, 3, 3)
sns.scatterplot(x=df['WindSpeed'], y=df['PowerConsumption_Zone1'])
plt.title("WindSpeed vs Power Consumption (Zone 1)")

plt.tight_layout()
plt.show()

# Correlation statistics
print("Correlation between Temperature and Power Consumption (Zone 1):",
      df['Temperature'].corr(df['PowerConsumption_Zone1']))
print("Correlation between Humidity and Power Consumption (Zone 1):",
      df['Humidity'].corr(df['PowerConsumption_Zone1']))
print("Correlation between WindSpeed and Power Consumption (Zone 1):",
      df['WindSpeed'].corr(df['PowerConsumption_Zone1']))
# Check for missing values
print(df.isnull().sum())

# Handle missing values using interpolation (suitable for time series)
df = df.interpolate(method='linear')  # Linear interpolation
df = df.ffill().bfill()  # Forward-fill and backward-fill

# Check for inconsistent data (e.g., outliers)
print(df.describe())

# Remove outliers (e.g., values outside 3 standard deviations)

z_scores = zscore(df.select_dtypes(include=['float64', 'int64']))
# Keep rows where all z-scores are less than 3
df = df[(z_scores < 3).all(axis=1)]

# Verify cleaned data
print(df.isnull().sum())
print(df.describe())
