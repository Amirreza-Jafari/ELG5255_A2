

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = 'Datasets/MCSDatasetNEXTCONLab.csv'  # Update with correct path
data = pd.read_csv(file_path)

# Define features and target
features = ['Latitude', 'Longitude', 'Day', 'Hour', 'Minute', 'Duration',
            'RemainingTime', 'Resources', 'Coverage', 'OnPeakHours', 'GridNumber']
target = 'Ligitimacy'

X = data[features]
y = data[target]

# Step 1: Clustering Step
# Apply K-Means to separate legitimate and fake tasks into clusters
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Latitude', y='Longitude', hue='Cluster', palette='viridis', alpha=0.7)
plt.title("Clustering of Tasks")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend(title="Cluster")
plt.show()

# Separate legitimate and fake clusters for Hybrid Approach
legitimate_clusters = data[data['Ligitimacy'] == 1]
fake_clusters = data[data['Ligitimacy'] == 0]

# Balance the dataset by using fake and legitimate clusters
balanced_data = pd.concat([legitimate_clusters, fake_clusters])
X_balanced = balanced_data[features]
y_balanced = balanced_data[target]

# Step 2: Hybrid Approach
# Split balanced dataset into train-test sets
X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(
    X_balanced, y_balanced, test_size=0.25, random_state=42
)

# Train a supervised model on the balanced data
rf_hybrid = RandomForestClassifier(random_state=42)
rf_hybrid.fit(X_train_hybrid, y_train_hybrid)
y_pred_hybrid = rf_hybrid.predict(X_test_hybrid)

# Step 3: Purely Supervised Model
# Apply SMOTE to balance the original dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split resampled dataset into train-test sets
X_train_supervised, X_test_supervised, y_train_supervised, y_test_supervised = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42
)

# Train a supervised model on the resampled data
rf_supervised = RandomForestClassifier(random_state=42)
rf_supervised.fit(X_train_supervised, y_train_supervised)
y_pred_supervised = rf_supervised.predict(X_test_supervised)

# Step 4: Evaluation
# Confusion matrices
cm_hybrid = confusion_matrix(y_test_hybrid, y_pred_hybrid)
cm_supervised = confusion_matrix(y_test_supervised, y_pred_supervised)

# Print confusion matrices
print("=== Hybrid Approach Confusion Matrix ===")
print(cm_hybrid)

print("\n=== Purely Supervised Confusion Matrix ===")
print(cm_supervised)

# Classification reports
print("\n=== Hybrid Approach Classification Report ===")
print(classification_report(y_test_hybrid, y_pred_hybrid, target_names=['Fake (0)', 'Legitimate (1)']))

print("\n=== Purely Supervised Classification Report ===")
print(classification_report(y_test_supervised, y_pred_supervised, target_names=['Fake (0)', 'Legitimate (1)']))
print('1- Reduced FN and FP in the Hybrid approach.\n 2- Better precision and recall for legitimate=0 and legitimate=1 tasks.')

# Plot confusion matrices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title("Hybrid Approach Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(cm_supervised, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Purely Supervised Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
