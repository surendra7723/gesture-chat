import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)


data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier(n_estimators=100, random_state=42)


# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Basic accuracy
accuracy = accuracy_score(y_test, y_predict)
print('Overall Accuracy: {:.2f}%'.format(accuracy * 100))

# Precision, Recall, and F1 Score
precision = precision_score(y_test, y_predict, average='weighted', zero_division=0)
recall = recall_score(y_test, y_predict, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_predict, average='weighted', zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Cross-validation scores
cv_scores = cross_val_score(model, data, labels, cv=5)
print("Cross-validation scores:", cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"CV Accuracy Standard Deviation: {cv_scores.std():.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_predict, zero_division=0))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_predict)

# Save the model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Feature Importance
if hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(20), feature_importance[sorted_idx])
    plt.title('Top 20 Feature Importances')
    plt.savefig('feature_importance.png')
    plt.show()
