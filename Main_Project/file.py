import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- 1. Load Data (Section 6: Data Used) ---
# We use the official Wisconsin Breast Cancer dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = Malignant (Cancer), 1 = Benign (Healthy)

print(f"Dataset Shape: {df.shape}")
print(df.head())

# --- 2. Preprocessing ---
# Split into Features (X) and Target (y)
X = df.drop('target', axis=1)
y = df['target']

# Split into Training and Testing sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. AI Algorithm (Section 5: AI Used) ---
# We will use Logistic Regression as per your project requirements
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Alternative: Support Vector Machine (SVM)
# model = SVC(kernel='linear') 
# model.fit(X_train, y_train)

# --- 4. Real-time Analysis & Prediction ---
y_pred = model.predict(X_test)

# --- 5. Performance Metrics (Section 4: Features) ---
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# --- Visualizing the Confusion Matrix ---
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Breast Cancer Detection Confusion Matrix')
plt.show()