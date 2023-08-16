import numpy as np
import cv2

# Read the image
image = cv2.imread('nono.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply morphological operations to remove small noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)

# Calculate the gradient magnitude
gradient = cv2.subtract(opening, cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel))

# Apply thresholding to create a binary image
_, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Apply morphological operations to remove small holes
sure_bg = cv2.dilate(binary, kernel, iterations=3)

# Find sure foreground (unknown region)
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Subtract sure foreground from sure background to get unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Label markers for the watershed algorithm
_, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0

# Apply watershed algorithm
markers = cv2.watershed(image, markers)

# Count the number of masks (excluding background)
number_of_masks = np.max(markers) - 1

# Label the marks with numbers and draw boundaries inside the areas
font = cv2.FONT_HERSHEY_SIMPLEX

for region_id in range(2, np.max(markers) + 1):  # Exclude background
    region_mask = markers == region_id
    region_area = np.sum(region_mask)
    if region_area > 100:  # Filter out small regions

        # Calculate contour area
        contours, _ = cv2.findContours(np.uint8(region_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = cv2.contourArea(contours[0])

        # Calculate convex perimeter
        convex_perimeter = cv2.arcLength(contours[0], True)

        # Calculate roundness
        roundness = (4 * np.pi * region_area) / (convex_perimeter * convex_perimeter)

        # Display roundness value at the center of the contour
        m = cv2.moments(np.uint8(region_mask))
        cX = int(m["m10"] / m["m00"])
        cY = int(m["m01"] / m["m00"])
        
        cv2.putText(image, f'Roundness: {roundness:.2f}', (cX - 100, cY), font, 0.5, (255, 0, 0), 2)

# Display the results
print(number_of_masks)
cv2.imshow('Roundness Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
