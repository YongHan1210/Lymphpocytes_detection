import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

class lympho_cell_detection:
    def __init__(self,image_path,file_path):
        self.image_path = image_path
        self.file_path = file_path
    
    def resize_image(self,image):  #resize image
        scale_percent = 30
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dsize = (width, height)
        image = cv2.resize(image, dsize)
        return image
    
    def RGB2CMYK(self,RGB_image):
        K = 1 - np.max(RGB_image,axis = 2)
        C = (1-RGB_image[...,2] - K)/(1-K)
        M = (1-RGB_image[...,1] - K)/(1-K)
        Y = (1-RGB_image[...,0] - K)/(1-K)
        CMYK_image= (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)
        return CMYK_image
    
    def CMYK2GRAY(self,CMYK_image):
        GRAY_image = cv2.cvtColor(CMYK_image, cv2.COLOR_BGR2GRAY)
        return GRAY_image

    def CMYK2GRB(self,c, m, y, k):
        r = 255 * (1 - c) * (1 - k)
        g = 255 * (1 - m) * (1 - k)
        b = 255 * (1 - y) * (1 - k)
        return np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)
    
    def GRAY_WHITEENHANCEMENT(self, GRAY_image):
        third_quartile = np.percentile(GRAY_image, 75)

        
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(1,1))
        ENHANCED_image = clahe.apply(np.uint8(np.clip((GRAY_image - third_quartile) * 255 / (255 - third_quartile), 0, 255)))
        ENHANCED_image[ENHANCED_image > third_quartile] = 255
        return ENHANCED_image

    def ZACK_ALGORITHM(self, GRAY_image, num_bins = 256): 

        # CITED from: https://www.mathworks.com/matlabcentral/fileexchange/28047-gray-image-thresholding-using-the-triangle-method?fbclid=IwAR3sAWs9hQNhi9oiI7vshz9ScLxlvIEx5a_ucxRuyPe8V4DAKGERQJyBvv0
        
        lehisto, _ = np.histogram(GRAY_image, bins=256, range=(0, 256), density=True)
        xmax = np.argmax(lehisto)
        xmax = round(np.mean(xmax))  # Can have more than a single value!
        # Find location of first and last non-zero values. # Values < h/10000 are considered zeros.
        h = lehisto[xmax]
        indi = np.where(lehisto > h / 10000)[0]
        fnz = indi[0]
        lnz = indi[-1]
        # Pick side as the side with the longer tail. Assume one tail is longer.
        lspan = xmax - fnz
        rspan = lnz - xmax
        if rspan > lspan:  # then flip lehisto
            lehisto = np.flip(lehisto)
            a = num_bins - lnz + 1
            b = num_bins - xmax + 1
            isflip = 1
        else:
            lehisto = lehisto
            isflip = 0
            a = fnz
            b = xmax
        # Compute parameters of the straight line from first non-zero to peak
        m = h / (b - a) 
        # Compute distances
        x1 = np.arange(0, b - a + 1)
        x1_adjusted = x1 + a
        x1_clipped = np.clip(x1_adjusted, 0, len(lehisto) - 1)
        y1 = lehisto[x1_clipped]
        beta = y1 + x1_clipped / m
        x2 = beta / (m + 1 / m)
        y2 = m * x2
        L = np.sqrt((y2 - y1)**2 + (x2 - x1_clipped)**2)
        # Obtain threshold as the location of maximum L.
        level = np.argmax(L)
        level = a + np.mean(level)
        # Flip back if necessary
        if isflip:
            level = num_bins - level + 1
        threshold = level / num_bins
        threshold_factor = 1
        increased_threshold = threshold * threshold_factor 
        binary_image = (GRAY_image > int(increased_threshold * 255 )).astype(np.uint8) * 255
        return binary_image
    
    def FILTERNOISE(self, binary_image):
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a mask and draw contours on the mask
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(binary_image)
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            #draw mask contour only when area>1000
            if area > 300:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        # Apply the mask to create the processed image
        drawc_image = binary_image.copy()
        drawc_image[mask == 255] = 0
        output_image = binary_image - drawc_image
        return output_image

    def watershed_segmentation(self,path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #cv2.imshow('gray', gray)
        
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel= np.ones((5, 5), np.uint8) , iterations=3)
        #cv2.imshow('opening', opening)
        gradient = cv2.subtract(opening, cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel = np.ones((3, 3), np.uint8) )) # Calculate the gradient magnitude
        #cv2.imshow('gradient', gradient)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Apply thresholding to create a binary image
       
        sure_bg = cv2.dilate(binary, kernel = np.ones((3, 3), np.uint8) , iterations=6) # Apply morphological operations to remove small holes
        #cv2.imshow('sure_bg', sure_bg)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 0) # Find sure foreground (unknown region)
        _, sure_fg = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg) 
        #cv2.imshow('sure_fg', sure_fg)
        sure_fg1 = cv2.erode(sure_fg,kernel = np.ones((2, 2), np.uint8),iterations =3)
        #cv2.imshow('sure_fg1', sure_fg1)
        unknown = cv2.subtract(sure_bg, sure_fg1) # Subtract sure foreground from sure background to get unknown region 
        #cv2.imshow('unknown', unknown)
        _, markers = cv2.connectedComponents(sure_fg1) # Label markers for the watershed algorithm
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers) # Apply watershed algorithm
       
        height, width = image.shape[:2]
        img = np.zeros((height, width, 3), dtype = np.uint8)
        imglist=[]
        numcount = 0
        for region_id in range(2, np.max(markers) + 1):  # Exclude background 
            region_mask = markers == region_id
            watershed_image = np.zeros_like(image)   
            imagecopy = image.copy()  # Mask out pixels outside the green contour
            imagecopy[~region_mask] = [0, 0, 0]
            img = watershed_image + imagecopy
            
            imglist.append(img)
            numcount+=1
            
        return imglist

    def watershed_segmentation1(self,path):
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray', gray)
        
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel= np.ones((5, 5), np.uint8) , iterations=3)
        #cv2.imshow('opening', opening)
        gradient = cv2.subtract(opening, cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel = np.ones((3, 3), np.uint8) )) # Calculate the gradient magnitude
        #cv2.imshow('gradient', gradient)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Apply thresholding to create a binary image
       
        sure_bg = cv2.dilate(binary, kernel = np.ones((3, 3), np.uint8) , iterations=6) # Apply morphological operations to remove small holes
        #cv2.imshow('sure_bg', sure_bg)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 0) # Find sure foreground (unknown region)
        _, sure_fg = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg) 
        #cv2.imshow('sure_fg', sure_fg)
        
        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area>200:
                sure_fg1 = cv2.erode(sure_fg,kernel = np.ones((2, 2), np.uint8),iterations =3)
                contours, _ = cv2.findContours(sure_fg1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_area = cv2.contourArea(contour)
                    if contour_area>150:
                        sure_fg1 = cv2.erode(sure_fg,kernel = np.ones((3, 3), np.uint8),iterations =3)
            else:
                sure_fg1 = sure_fg
        
        #cv2.imshow('sure_fg1', sure_fg1)
        unknown = cv2.subtract(sure_bg, sure_fg1) # Subtract sure foreground from sure background to get unknown region 
        #cv2.imshow('unknown', unknown)
        _, markers = cv2.connectedComponents(sure_fg1) # Label markers for the watershed algorithm
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers) # Apply watershed algorithm
        
       
        height, width = image.shape[:2]
        img = np.zeros((height, width, 3), dtype = np.uint8)
        imglist=[]
        numcount = 0
        for region_id in range(2, np.max(markers) + 1):  # Exclude background 
            region_mask = markers == region_id
            watershed_image = np.zeros_like(image)   
            imagecopy = image.copy()  # Mask out pixels outside the green contour
            imagecopy[~region_mask] = [0, 0, 0]
            img = watershed_image + imagecopy
            #img = cv2.dilate(img,kernel= np.ones((3, 3), np.uint8),iterations = 1)
            imglist.append(img)
            
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            numcount+=1
        return numcount, imglist

    def GROUPCELL_DIFFERENTAITE(self,image,thres):
        CELL_LESSTHAN_THRES_list =[]
        CELL_MORETHAN_THRES_list =[]

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            
            contour_area = cv2.contourArea(contour)

            if contour_area < thres:
                result_less_than_thres = np.zeros_like(image)
                cv2.drawContours(result_less_than_thres, [contour], -1, (255, 255, 255), -1)
                masked_imageresult_less_than_thres = cv2.bitwise_and(image, result_less_than_thres)
                CELL_LESSTHAN_THRES_list.append(masked_imageresult_less_than_thres)

            else:
                result_more_than_thres = np.zeros_like(image)
                cv2.drawContours(result_more_than_thres, [contour], -1, (255, 255, 255), -1)
                masked_imageresult_more_than_thres = cv2.bitwise_and(image, result_more_than_thres)
                CELL_MORETHAN_THRES_list.append(masked_imageresult_more_than_thres)

        return CELL_LESSTHAN_THRES_list,CELL_MORETHAN_THRES_list



    def cell_detection(self):

        '''LIST define'''
        cell_detected_list = []            # individual detected cell list where each component will be an image storing an individual cell
        cell_detected_mask_list = []       # individual detected cell list where each component will be an image storing an individual masked with original colored cell
        cell_detected_variable_list =[]    

        '''Original Image Processing'''
        original_image = cv2.imread(self.image_path)             # read image path
        original_image = self.resize_image(original_image)       # resize image 
        height, width, _ = original_image.shape                  # get height and width of image
        cv2.imshow('Original Image', original_image)
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\Original_Image.jpeg', original_image)
     
        '''CMYK Image Processing'''
        CMYK_image = self.RGB2CMYK(original_image)               # convert RGB image to CMYK image
        cyan_channel = CMYK_image[:, :, 0] * 0.5                 # get only cyan channel of CMYK      # multiplify by 0.5 reduce the overall intensity level which then thresholding will be more efficient
        magenta_channel = CMYK_image[:, :, 1]                    # get only magenta channel of CMYK
        yellow_channel = CMYK_image[:, :,2]                      # get only yellow channel of CMYK
        cv2.imshow('CMYK Image', CMYK_image)
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\CMYK_Image.jpeg', CMYK_image)
        cv2.imshow('Cyan Channel CMYK Image', cyan_channel)
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\Cyan_CMYK_Image.jpeg', cyan_channel)
        cv2.imshow('Magenta Channel CMYK Image', magenta_channel) 
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\Magenta_CMYK_Image.jpeg', magenta_channel)
        cv2.imshow('Yellow Channel CMYK Image', yellow_channel)
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\Yellow_CMYK_Image.jpeg', yellow_channel)
        
        '''ZACK ALGORITHM Processing'''
        ZACK_ALG_image = self.ZACK_ALGORITHM(cyan_channel)      # perform ZACK ALGORITHM using cyan channel which can differentiate the cell more effectively
        cv2.imshow('ZACK_ALG_image', ZACK_ALG_image) 
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\Zack_Alg_Image.jpeg', ZACK_ALG_image)

        '''FILTERING & CLEANING ZACK ALGORITHM IMAGE'''
        FILTERNOISE_ZACK_ALG_image = self.FILTERNOISE(ZACK_ALG_image)      # perform filtering and cleaning ZACK ALGORITHM binary image
        cv2.imshow('NOISE_FILTERED ZACK ALG Image', FILTERNOISE_ZACK_ALG_image) 
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\FILTER_Zack_Alg_Image.jpeg', FILTERNOISE_ZACK_ALG_image)
        
        '''DIFFERENTIATE GROUPED & INDIVIDUAL CELL'''
        CELL_LESSTHAN_THRES_list, CELL_MORETHAN_THRES_list = self.GROUPCELL_DIFFERENTAITE(FILTERNOISE_ZACK_ALG_image, 1400)     # grouped as group cell if the area of contour is greater than 1400   # return list which separate each contour as an image component
        LESSTHAN_THRES_image = cv2.cvtColor(np.zeros_like(original_image), cv2.COLOR_BGR2GRAY)       # create a black image 
        MORETHAN_THRES_image = cv2.cvtColor(np.zeros_like(original_image), cv2.COLOR_BGR2GRAY)

        '''INDIVIDUAL CELL'''
        for image in CELL_LESSTHAN_THRES_list:        
            LESSTHAN_THRES_image = LESSTHAN_THRES_image + image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  
            cell_detected_list.append(image)
        cv2.imshow('CELL LESSTHAN THRES image', LESSTHAN_THRES_image) 
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\CELL_LESSTHAN_THRES_Image.jpeg', LESSTHAN_THRES_image)

        '''GROUPED CELL'''
        for image in CELL_MORETHAN_THRES_list:
            MORETHAN_THRES_image = MORETHAN_THRES_image + image
        cv2.imshow('CELL MORETHAN THRES image', MORETHAN_THRES_image)
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\CELL_MORETHAN_THRES_Image.jpeg', MORETHAN_THRES_image)

        '''WATERSHED SEGMENTATION FOR GROUPED CELL'''
        WATERSHED_SEGMENTATION_image = np.zeros_like(original_image)     # create black image for overall segmentation result
        WATERSHED_SEGMENTATION_1_image = np.zeros_like(original_image)   # create black image for segmentation 1  (FIRST SEGMENTATION)
        WATERSHED_SEGMENTATION_2_image = np.zeros_like(original_image)   # create black image for segmentation 2  (SECOND SEGMENTATION)

        for group_cell in CELL_MORETHAN_THRES_list:                      # for each grouped_cell image stored in the grouped cell list CELL_MORETHAN_THRES_list
            cv2.imwrite("output_image.jpeg", group_cell)                 # save image as output_image.jpeg which will then be read to obtain rgb image
            watershed_image_list = self.watershed_segmentation("output_image.jpeg")        # return FIRST ITERATION SEGMENTATION image result list 
            for segmented_image in watershed_image_list:                                   # for result image in FIRST ITERATION SEGMENTATION list
                segmented_image = segmented_image.astype(np.uint8)                          
                segmented_image = cv2.cvtColor(segmented_image,cv2.COLOR_BGR2GRAY) 
                ret, segmented_image = cv2.threshold(segmented_image, 127, 255, 0)
                contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       # find the area of the segmented cell

                if contours:
                    contour = contours[0]
                    contour_area = cv2.contourArea(contour)
                    if contour_area == 0:
                        cv2.imwrite("error.jpeg", segmented_image)
                    if contour_area > 1400:                                                                      # if segmented cell area > 1400 which means the FIRST ITERATION SEGMENTATION fails and will perfrom SECOND ITERATION SEGEMNTATION
                        cv2.imwrite("output_image1.jpeg", segmented_image)
                        numcount, watershed_image_list1 = self.watershed_segmentation1("output_image1.jpeg")
                        for segmented_image1 in watershed_image_list1:
                            cell_detected_list.append(segmented_image1)
                            WATERSHED_SEGMENTATION_image = WATERSHED_SEGMENTATION_image + segmented_image1 
                            WATERSHED_SEGMENTATION_2_image = WATERSHED_SEGMENTATION_2_image + segmented_image1
                    else:
                        if len(segmented_image.shape) < 3:
                            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
                        cell_detected_list.append(segmented_image)
                        WATERSHED_SEGMENTATION_image = WATERSHED_SEGMENTATION_image + segmented_image 
                        WATERSHED_SEGMENTATION_1_image = WATERSHED_SEGMENTATION_1_image + segmented_image 

        cv2.imshow('WATERSHED SEGMENTATION image', WATERSHED_SEGMENTATION_image ) 
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\WATERSHED_SEGMENTATION_Image.jpeg', WATERSHED_SEGMENTATION_image)
        cv2.imshow('WATERSHED SEGMENTATION 1 image', WATERSHED_SEGMENTATION_1_image ) 
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\WATERSHED_SEGMENTATION_1_Image.jpeg', WATERSHED_SEGMENTATION_1_image)
        cv2.imshow('WATERSHED SEGMENTATION 2 image', WATERSHED_SEGMENTATION_2_image )
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\WATERSHED_SEGMENTATION_2_Image.jpeg', WATERSHED_SEGMENTATION_2_image)
        
        data_list = []
        with open(self.file_path, 'r') as file:  
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Check if the line is not empty
                    x, y = map(int, line.split())
                    data_list.append((x, y))
        
        image = np.zeros((height,width, 3), dtype=np.uint8)
        for x,y in data_list:
            x = int(x*0.3)
            y = int(y*0.3)
            cv2.circle(image, (x,y), 20, (0, 0, 255), -1) 
        cv2.imshow('Red Circle on Black Image', image) 
        binary_image = cv2.imread('D:\Image_procesing\Lymphpocytes_detection\RESULT\FILTER_Zack_Alg_Image.jpeg')
        num_count = 0
        FINAL_RESULT_image = np.zeros_like(original_image)
        for result_cell_img in cell_detected_list:
            result_cell_img = cv2.bitwise_and(result_cell_img,binary_image)
            
            MASKED_image = cv2.bitwise_and(original_image, result_cell_img)
            cell_detected_mask_list.append(MASKED_image)
            FINAL_RESULT_image = cv2.add(FINAL_RESULT_image , result_cell_img)  
            result_cell_img = cv2.cvtColor(result_cell_img, cv2.COLOR_BGR2GRAY)  
            contours, _ = cv2.findContours(result_cell_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                num_count += 1
                
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                
                convex_hull = cv2.convexHull(contour)
                convex_area = cv2.contourArea(convex_hull)
                convex_perimeter = cv2.arcLength(convex_hull, True)
                if len(contour) >= 5:
                    ellipse = cv2.fitEllipse(contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                if convex_perimeter != 0 and convex_area != 0:
                    roundness = (4 * math.pi * area) / (convex_perimeter ** 2)
                    solidity = area / convex_area
                    elongation = 1 - (minor_axis / major_axis)
                    eccentricity = math.sqrt(major_axis**2 - minor_axis**2) / major_axis
                    convexity = convex_perimeter / perimeter
                else:
                    roundness = solidity = elongation = eccentricity = convexity = 0
                # Display roundness on the mask
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                
            

            
            cell_detected_variable_list.append(((cX,cY),convex_area,roundness,solidity,elongation,eccentricity,convexity))

        cv2.imshow("FINAL_RESULT_image",FINAL_RESULT_image)
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\FINAL_RESULT_Image.jpeg', FINAL_RESULT_image)

        FINAL_RESULT_MASKED_image = cv2.bitwise_and(original_image, FINAL_RESULT_image)
        cv2.imshow('FINAL RESULT MAKSED image', FINAL_RESULT_MASKED_image)    
        cv2.imwrite('D:\Image_procesing\Lymphpocytes_detection\RESULT\FINAL_RESULT_MASKED_Image.jpeg', FINAL_RESULT_MASKED_image)


        
            

        #cv2.waitKey(0)
        cv2.destroyAllWindows
        return cell_detected_variable_list
        

        


if __name__ == "__main__":
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im003_1.jpg
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im034_0.jpg
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im056_1.jpg
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im059_1.jpg
    loaded_svm_classifier = joblib.load('svm_classifier_model.joblib')
    print("Trained model loaded")

    image_path = "D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im003_1.jpg"
    file_path = r'D:\\Image_procesing\\Ass\\ALL_IDB1\\ALL_IDB1\\xyc\\Im003_1.xyc'

    lcd = lympho_cell_detection(image_path,file_path)

    cell_detected_variable_list = lcd.cell_detection()

    data_list = []
    with open(file_path, 'r') as file:  
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Check if the line is not empty
                x, y = map(int, line.split())
                data_list.append((int(x*0.3), int(y*0.3)))
    
    len_data_list = len(data_list)
    Correct_numcount = 0
    
    for (cX,cY),area,roundness,solidity,elongation,eccentricity,convexity in cell_detected_variable_list:
        
        
        Prediction_flag = False
        Cell_detect_flag = False
        

        for (x,y) in data_list:
            if (abs(cX-x)<5) and (abs(cY-y)<5):
                #print(f"This cell is correct cXcY={cX},{cY}   xy={x},{y}")
                
                data_list.remove((x,y))
                Cell_detect_flag = True
        

        new_test_data = [[roundness, solidity, elongation, eccentricity, convexity]]    
        new_test_predictions = loaded_svm_classifier.predict(new_test_data)
        #print("Predictions on new test data:", new_test_predictions)
        Prediction_flag = True if new_test_predictions[0] == 1 else False
        
        if Cell_detect_flag:        
            print(f"PF:{Prediction_flag}   XYCCentroid: {int(x/0.3)},{int(y/0.3)}   Centroid: {int(cX/0.3)},{int(cY/0.3)}  Area: {area}    Roundness:{roundness:.2f}   Solidty:{solidity:.2f}   Elongation:{elongation:.2f}    Eccentricity:{eccentricity:.2f}   Convexity:{convexity:.2f}")    
        else:
            print(f"PF:{Prediction_flag}   XYCCentroid: None   Centroid: {int(cX/0.3)},{int(cY/0.3)}  Area: {area}    Roundness:{roundness:.2f}   Solidty:{solidity:.2f}   Elongation:{elongation:.2f}    Eccentricity:{eccentricity:.2f}   Convexity:{convexity:.2f}")      

        if Prediction_flag == True and Cell_detect_flag == True:
            
            Correct_numcount = Correct_numcount + 1
    
    if len_data_list == 0:
        percentage_ratio = 100 if Correct_numcount==0 and len_data_list == 0 else 0
    else:
        percentage_ratio = Correct_numcount/len_data_list *100
    print("_"*10 + "RESULT" +"_"*10)
    print(f"XYC CELL COUNT = {len_data_list}    DETECTED CELL COUNT = {Correct_numcount}")
    print(f"Pecentage Ratio = {percentage_ratio:.2f}")
