import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

class lympho_cell_detection:
    def __init__(self,image_path):
        self.image_path = image_path
    
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
    
    def FILTER_SMALLNOISE(self, binary_image):
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

    def watershed_segmentation(self):
        image = cv2.imread("output_image.jpeg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray', gray)

        #watershed segmentation to separate cell
        kernel = np.ones((3, 3), np.uint8)
        
        
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel= np.ones((5, 5), np.uint8) , iterations=3)
        #cv2.imshow('opening', opening)
        gradient = cv2.subtract(opening, cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel )) # Calculate the gradient magnitude
        #cv2.imshow('gradient', gradient)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Apply thresholding to create a binary image
       
        sure_bg = cv2.dilate(binary, kernel , iterations=6) # Apply morphological operations to remove small holes
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
       
        font = cv2.FONT_HERSHEY_SIMPLEX # Label the marks with numbers and draw boundaries inside the areas
        height, width = image.shape[:2]
        img = np.zeros((height, width, 3), dtype = np.uint8)
        for region_id in range(2, np.max(markers) + 1):  # Exclude background 
            region_mask = markers == region_id
            region_area = np.sum(region_mask)
            
            if region_area > 100:  # Filter out small regions+
                

               
                imagecopy = image.copy()  # Mask out pixels outside the green contour
                imagecopy[~region_mask] = [0, 0, 0]
                img = img + imagecopy
        #cv2.imshow('img', img)
        return img

    def differentiate_grouped_lymphocyte(self,image):
        result_less_than_1100 = np.zeros_like(image)
        result_more_than_1100 = np.zeros_like(image)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            
            contour_area = cv2.contourArea(contour)
            if contour_area < 1100:
                cv2.drawContours(result_less_than_1100, [contour], -1, (255, 255, 255), -1)
            else:
                cv2.drawContours(result_more_than_1100, [contour], -1, (255, 255, 255), -1)
        return result_less_than_1100,result_more_than_1100



    def cell_detection(self):

        # read cell_image_path
        original_image = cv2.imread(self.image_path)
        original_image = self.resize_image(original_image)
        #cv2.imshow('original_image', original_image)
     
        CMYK_image = self.RGB2CMYK(original_image)
        

        cyan_channel = CMYK_image[:, :, 0] *0.5
        magenta_channel = CMYK_image[:, :, 1]
        yellow_channel = CMYK_image[:, :,2]
        #cv2.imshow('Cyan Channel', cyan_channel)
        # cv2.imshow('Magenta Channel', magenta_channel)     
        # cv2.imshow('yellow_channel', yellow_channel)   

        
  

        GRAY_image = self.CMYK2GRAY(CMYK_image)
        
        ZACK_ALG_image_Cyan_Channel = self.ZACK_ALGORITHM(cyan_channel)
        #cv2.imshow('ZACK_ALG_image_Cyan_Channel', ZACK_ALG_image_Cyan_Channel) 
        # kernel_size = 5  # You can adjust this value to change the filter size
        # filtered_image = cv2.medianBlur(ZACK_ALG_image_Cyan_Channel, kernel_size)
        #cv2.imshow('filtered_image', filtered_image) 
        smallNOISE_FILTERED_imagefiltered = self.FILTER_SMALLNOISE(ZACK_ALG_image_Cyan_Channel)
        #cv2.imshow('smallNOISE_FILTERED_imagefiltered', smallNOISE_FILTERED_imagefiltered) 
        

        
        result_less_than_1100, result_more_than_1100 = self.differentiate_grouped_lymphocyte(smallNOISE_FILTERED_imagefiltered)
        #cv2.imshow('Contours Less Than 1000', result_less_than_1100)
        
        masked_imageresult_more_than_1100 = cv2.bitwise_and(smallNOISE_FILTERED_imagefiltered, result_more_than_1100)
        masked_imageresult_less_than_1100 = cv2.bitwise_and(smallNOISE_FILTERED_imagefiltered, result_less_than_1100)
        #cv2.imshow('masked_imageresult_more_than_1100', masked_imageresult_more_than_1100)
        #cv2.imshow('masked_image', masked_imageresult_more_than_1100)
        cv2.imwrite("output_image.jpeg", masked_imageresult_more_than_1100)





 
        
        
        watershed_image = self.watershed_segmentation()
        watershed_image = cv2.cvtColor(watershed_image,cv2.COLOR_BGR2GRAY)


        
        
    
        


        combine_image = cv2.bitwise_or(watershed_image,masked_imageresult_less_than_1100)
        cv2.imshow('combine_image', combine_image)
        
        
        three_channel_image = np.stack((combine_image,) * 3, axis=-1)
        print(three_channel_image.shape)
        print(original_image.shape)
        masked_image = cv2.bitwise_and(original_image, three_channel_image)
        cv2.imshow('masked_image', masked_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows
        

        


if __name__ == "__main__":
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im003_1.jpg
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im034_0.jpg
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im056_1.jpg
    #D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im059_1.jpg
    path = "D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im006_1.jpg"

    lcd = lympho_cell_detection(path)

    lcd.cell_detection()
        

    
    

    

'''contours, _ = cv2.findContours(np.uint8(region_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = contours[0]
                    
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    
                    convex_hull = cv2.convexHull(contour)
                    convex_area = cv2.contourArea(convex_hull)
                    convex_perimeter = cv2.arcLength(convex_hull, True)
                    ellipse = cv2.fitEllipse(contour)
                    major_axis = max(ellipse[1])
                    minor_axis = min(ellipse[1])

                    roundness = (4 * math.pi * area) / (convex_perimeter ** 2)
                    solidity = area / convex_area
                    elongation = 1 - (minor_axis / major_axis)
                    eccentricity = math.sqrt(major_axis**2 - minor_axis**2) / major_axis
                    convexity = convex_perimeter / perimeter

                    # Display roundness on the mask
                    # roundness_text = f" {number_of_masks}"
                    # text_position = (int(contour[0][0][0]), int(contour[0][0][1]) )
                    # cv2.putText(img, roundness_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    print(f"Mask:{number_of_masks}  Area: {area}    Roundness:{roundness:.2f}   Solidty:{solidity:.2f}   Elongation:{elongation:.2f}    Eccentricity:{eccentricity:.2f}   Convexity:{convexity:.2f}")'''
