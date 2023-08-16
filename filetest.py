import os
from maincode import main_function
if __name__ == "__main__":
    
 
    
    xycpath = r"D:\\Image_procesing\\Ass\\ALL_IDB1\\ALL_IDB1\\xyc"
    overall_xycfilename = os.listdir(xycpath)
    total_cell_recorded_array=[]
    for file_name in overall_xycfilename:
        filepath = os.path.join(xycpath, file_name)

        if file_name[6] == "0":
            total_cell_recorded = 0
        else:
            with open(filepath,'r') as file:
                li = file.readlines()
                total_cell_recorded = len(li)
        total_cell_recorded_array.append(total_cell_recorded)
        

    imgpath = r"D:\\Image_procesing\\Ass\\ALL_IDB1\\ALL_IDB1\\im"
    overall_imgfilename = os.listdir(imgpath)
    total_cell_detected_array = []
    for file_name in overall_imgfilename:
        filepath = os.path.join(imgpath, file_name)
        
        num_cell_detected = main_function(filepath)
        total_cell_detected_array.append(num_cell_detected)

    num = 1
    while num<109:
        if total_cell_recorded_array[num-1] == 0:
            total_cell_file = 0
        else:
            total_cell_file = 1
        diff = total_cell_recorded_array[num-1] - total_cell_detected_array[num-1]
        print(f"Number of lines in the notepad file Im{num:03d}_{total_cell_file}.xyc:{total_cell_recorded_array[num-1]} Number of cells detected: {total_cell_detected_array[num-1]} difference: {diff}")
        num+=1


    