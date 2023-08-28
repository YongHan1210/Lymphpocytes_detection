import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from LCD import lympho_cell_detection
 
if __name__ == "__main__":
    


    imgpath = r"D:\\Image_procesing\\Ass\\ALL_IDB2\\ALL_IDB2\\img"  
    img_filename = os.listdir(imgpath)

    test_dataset_X = []
    test_dataset_y = []

    for imgfile in img_filename:
        img_path = os.path.join(imgpath, imgfile)
        #print(img_path)
        statement = int(imgfile[6])
        
        lcd = lympho_cell_detection(img_path) 
        roundness,solidity,elongation,eccentricity,convexity = lcd.cell_detection()
        #print(f" Roundness:{roundness:.2f}   Solidty:{solidity:.2f}   Elongation:{elongation:.2f}    Eccentricity:{eccentricity:.2f}   Convexity:{convexity:.2f}")
        if roundness != 0 and solidity != 0 and elongation != 0 and eccentricity != 0 and convexity != 0:
            test_dataset_X.append([roundness,solidity,elongation,eccentricity,convexity])
            test_dataset_y.append([statement])

    # print(len(test_dataset_X))
    #print(test_dataset_X)
    # for test_dataset_X_data, test_dataset_y_data in zip(test_dataset_X, test_dataset_y):
    #     print(test_dataset_y_data, test_dataset_X_data)
    test_dataset_y = np.ravel(test_dataset_y)
    num_samples = len(test_dataset_X)
    num_features = 5
    

    X_train, X_test, y_train, y_test = train_test_split(test_dataset_X, test_dataset_y, test_size=0.05, random_state=60)

    # Initialize the SVM classifier
    svm_classifier = SVC(kernel='linear')  # You can try other kernels like 'rbf' as well

    # Train the SVM classifier on the training data
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = svm_classifier.predict(X_test)
    print(X_test)
    
    # Calculate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    model_filename = 'svm_classifier_model.joblib'
    joblib.dump(svm_classifier, model_filename)
    print(f"Trained model saved as {model_filename}")

    # Load the trained classifier from the file
    loaded_svm_classifier = joblib.load(model_filename)
    print("Trained model loaded")

    # You can now use the loaded classifier to make predictions on new test data
    new_test_data = [[0.9632046697632676, 0.9780526735833999, 0.051938067765762486, 0.3180857944773453, 0.9486012111153495]]
    new_test_predictions = loaded_svm_classifier.predict(new_test_data)
    print("Predictions on new test data:", new_test_predictions)



        
         
    