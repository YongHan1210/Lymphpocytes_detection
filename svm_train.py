

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

np.random.seed(42)
num_samples = 10000
num_features = 6
# Generate or load your dataset and labels
X = np.random.rand(num_samples, num_features)  # Replace with your dataset
print(X)
y = np.random.choice([0, 1], size=num_samples)  # Replace with your labels

# Initialize the SVM classifier
svm_classifier = SVC(kernel='linear')  # You can try other kernels like 'rbf' as well

# Perform k-fold cross-validation (e.g., k=5)
num_folds = 3
scores = cross_val_score(svm_classifier, X, y, cv=num_folds, scoring='accuracy')

# Print accuracy scores for each fold
for fold_idx, accuracy in enumerate(scores, start=1):
    print(f"Fold {fold_idx}: Accuracy = {accuracy:.2f}")

# Calculate and print the mean accuracy across all folds
mean_accuracy = np.mean(scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
