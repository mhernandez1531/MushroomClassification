import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Loading dataset
column_names = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
df = pd.read_csv('/Users/marianahernandez/Downloads/agaricus-lepiota.data', header=None, names=column_names)

# Data preprocessing
le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

# Model Tuning
n_neighbors_values = [3, 5, 7, 9]
for n in n_neighbors_values:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for n_neighbors={n}: {accuracy:.2f}")

distance_metrics = ['euclidean', 'manhattan', 'minkowski']
for metric in distance_metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {metric} distance: {accuracy:.2f}")

# Cross-Validation and Overfitting Check
cv_scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
print("\nCross-validation accuracy scores:", cv_scores)
print("Average cross-validation accuracy:", cv_scores.mean())

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Poisonous', 'Edible'], yticklabels=['Poisonous', 'Edible'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Comparison Bar Chart (assuming you have results for all models)
models = ['KNN', 'Decision Tree', 'Random Forest']
accuracies = [accuracy, dt_accuracy, rf_accuracy]  # Replace with actual accuracy values
precision_scores = [precision, dt_precision, rf_precision]  # Replace with actual precision values
recall_scores = [recall, dt_recall, rf_recall]  # Replace with actual recall values
f1_scores = [f1, dt_f1, rf_f1]  # Replace with actual F1 scores

bar_width = 0.2
index = range(len(models))

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(index, accuracies, bar_width, label='Accuracy')
bar2 = ax.bar([i + bar_width for i in index], precision_scores, bar_width, label='Precision')
bar3 = ax.bar([i + 2 * bar_width for i in index], recall_scores, bar_width, label='Recall')
bar4 = ax.bar([i + 3 * bar_width for i in index], f1_scores, bar_width, label='F1 Score')

ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Comparison of KNN, Decision Tree, and Random Forest Models')
ax.set_xticks([i + 1.5 * bar_width for i in index])
ax.set_xticklabels(models)
ax.legend()
plt.show()
