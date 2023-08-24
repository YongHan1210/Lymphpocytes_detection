import cv2
import os
from LymphoCellDetection import lympho_cell_detection
 
if __name__ == "__main__":
    

    imgpath = r"D:\\Image_procesing\\Ass\\ALL_IDB1\\ALL_IDB1\\im" 
    overall_imgfilename = os.listdir(imgpath)
    total_cell_detected_array = []
    for file_name in overall_imgfilename:
        filepath = os.path.join(imgpath, file_name)
        print(filepath)
        lcd = lympho_cell_detection(filepath) 
        lcd.cell_detection()
        

          
    