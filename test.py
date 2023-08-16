import cv2
import numpy as np


image = cv2.imread("output_image.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#watershed segmentation to separate cell
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2) # Apply morphological operations to remove small noise
gradient = cv2.subtract(opening, cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)) # Calculate the gradient magnitude
_, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Apply thresholding to create a binary image
sure_bg = cv2.dilate(binary, kernel, iterations=3) # Apply morphological operations to remove small holes
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5) # Find sure foreground (unknown region)
_, sure_fg = cv2.threshold(dist_transform, 0.6*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg) 
unknown = cv2.subtract(sure_bg, sure_fg) # Subtract sure foreground from sure background to get unknown region 
_, markers = cv2.connectedComponents(sure_fg) # Label markers for the watershed algorithm
markers += 1
markers[unknown == 255] = 0
markers = cv2.watershed(image, markers) # Apply watershed algorithm
number_of_masks = 0
font = cv2.FONT_HERSHEY_SIMPLEX # Label the marks with numbers and draw boundaries inside the areas
height, width = image.shape[:2]
img = np.zeros((height, width, 3), dtype = np.uint8)
for region_id in range(2, np.max(markers) + 1):  # Exclude background 
    region_mask = markers == region_id
    region_area = np.sum(region_mask)
    if region_area > 100:  # Filter out small regions
        contours, _ = cv2.findContours(np.uint8(region_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(contours[0])
        convex_perimeter = cv2.arcLength(contours[0], True)# Calculate convex perimeter
        roundness = (4 * np.pi * region_area) / (convex_perimeter * convex_perimeter)# Calculate roundness
        if roundness> 0.70:
            number_of_masks +=1
            imagecopy = image.copy()  # Mask out pixels outside the green contour
            imagecopy[~region_mask] = [0, 0, 0]
            img = img + imagecopy
cv2.imshow('Segmented Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows