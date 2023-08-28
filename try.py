from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import os
from LCD import lympho_cell_detection

imgpath = r"D:\\Image_procesing\\Ass\\ALL_IDB2\\ALL_IDB2\\img"
img_filename = os.listdir(imgpath)

test_dataset_X = []
test_dataset_y = []

for imgfile in img_filename:
    img_path = os.path.join(imgpath, imgfile)
    statement = int(imgfile[6])
    
    lcd = lympho_cell_detection(img_path)
    roundness, solidity, elongation, eccentricity, convexity = lcd.cell_detection()
    
    if roundness != 0 and solidity != 0 and elongation != 0 and eccentricity != 0 and convexity != 0:
        test_dataset_X.append([roundness, solidity, elongation, eccentricity, convexity])
        test_dataset_y.append(statement)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(test_dataset_X, test_dataset_y, test_size=0.1, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Transform only the features, not the labels

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm_classifier = SVC()
grid_search = GridSearchCV(svm_classifier, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Perform k-fold cross-validation
num_folds = 9  # You can adjust this value
scores = cross_val_score(svm_classifier, X_train_scaled, y_train, cv=num_folds)

# Print the accuracy for each fold
for fold, score in enumerate(scores, start=1):
    print(f"Fold {fold} Accuracy: {score:.2f}")

# Calculate and print the average accuracy across folds
average_accuracy = scores.mean()
print(f"Average Accuracy: {average_accuracy:.2f}")
