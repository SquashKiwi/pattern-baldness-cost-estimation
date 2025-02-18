import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv("C:/Users/teega/Downloads/bald_final_data (1).csv")

# Encode the 'race' column using Label Encoding
label_encoder = LabelEncoder()
data['race_encoded'] = label_encoder.fit_transform(data['race'])

# Select the relevant columns for correlation
columns_to_include = ['Class of Baldness (1,2,3,4)', 'age', 'smoking', 'race_encoded']
correlation_matrix = data[columns_to_include].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap (Class, Age, Smoking, Race)')
plt.show()
