import cv2
import os
import joblib

from sklearn.linear_model import SGDClassifier
import numpy as np
from LymphoCellDetection import lympho_cell_detection
 
if __name__ == "__main__":
    

    imgpath = r"D:\\Image_procesing\\Ass\\ALL_IDB1\\ALL_IDB1\\im"  
    filepath = r'D:\\Image_procesing\\Ass\\ALL_IDB1\\ALL_IDB1\\xyc'
    overall_imgfilename = os.listdir(imgpath)
    overall_filefilename = os.listdir(filepath)
    print("Trained model loaded")
    svm_classifier = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
    model_filename = 'svm_classifier_model.joblib'
    loaded_svm_classifier = joblib.load(model_filename)

    overall_file_num = 108
    overall_percentage_sum = 0
    overall_xyc_cell_count = 0
    overall_detected_cell_count = 0



    for img_file_name, file_file_name in zip(overall_imgfilename, overall_filefilename):
        img_path = os.path.join(imgpath, img_file_name)
        file_path = os.path.join(filepath, file_file_name)
        print(img_path,file_path)

        lcd = lympho_cell_detection(img_path,file_path)
        cell_detected_variable_list = lcd.cell_detection()

        data_list = []
        with open(file_path, 'r') as file:  
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Check if the line is not empty
                    x, y = map(int, line.split())
                    data_list.append((int(x*0.3), int(y*0.3)))
                    overall_xyc_cell_count += 1
        
        len_data_list = len(data_list)
        Correct_numcount = 0
        
        
        for (cX,cY),area,roundness,solidity,elongation,eccentricity,convexity in cell_detected_variable_list:
            
            
            Prediction_flag = False
            Cell_detect_flag = False
            

            for (x,y) in data_list:
                if (abs(cX-x)<5) and (abs(cY-y)<5):
                    print(f" Centroid: {cX},{cY}  Area: {area}    Roundness:{roundness:.2f}   Solidty:{solidity:.2f}   Elongation:{elongation:.2f}    Eccentricity:{eccentricity:.2f}   Convexity:{convexity:.2f}")
                    data_list.remove((x,y))
                    Cell_detect_flag = True


            new_test_data = [[roundness, solidity, elongation, eccentricity, convexity]]    
            new_test_predictions = loaded_svm_classifier.predict(new_test_data)
            #print("Predictions on new test data:", new_test_predictions)
            Prediction_flag = True if new_test_predictions[0] == 1 else False

            if Prediction_flag == True and Cell_detect_flag == True:
                Correct_numcount = Correct_numcount + 1
                overall_detected_cell_count +=1
                # correct_label = 1

                # # Retrain the SVM classifier on the updated training data
                # svm_classifier.partial_fit(new_test_data, [correct_label], classes=np.unique([0, 1]))
                # print("Retrained SVM classifier")

                # # Save the updated classifier
                # joblib.dump(svm_classifier, model_filename)
                # print(f"Updated model saved as {model_filename}")
        
        if len_data_list == 0:
            percentage_ratio = 100 if Correct_numcount==0 and len_data_list == 0 else 0
        else:
            percentage_ratio = Correct_numcount/len_data_list *100
        print("_"*10 + "RESULT" +"_"*10)
        print(f"XYC CELL COUNT = {len_data_list}    DETECTED CELL COUNT = {Correct_numcount}")
        print(f"Pecentage Ratio = {percentage_ratio:.2f}")
        
        overall_percentage_sum = overall_percentage_sum + percentage_ratio


            
    
    overall_percentage_ratio = overall_percentage_sum/overall_file_num
    print("\n\n")
    print("*"*10 + "OVERALL DATA" + "*"*10)
    print(f"XYC CELL COUNT = {overall_xyc_cell_count}    DETECTED CELL COUNT = {overall_detected_cell_count}")
    print(f"Overall Percentage Ratio = {overall_percentage_ratio:.2f} %")
