import cv2
import numpy as np



def resizeimg(img):  #resize img
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    img = cv2.resize(img, dsize)
    return img

def main_function(path):
    oriimg = cv2.imread(path)
    oriimg = resizeimg(oriimg)


    #RGB to CMYK
    K = 1 - np.max(oriimg,axis = 2)
    C = (1-oriimg[...,2] - K)/(1-K)
    M = (1-oriimg[...,1] - K)/(1-K)
    Y = (1-oriimg[...,0] - K)/(1-K)
    CMYK_img= (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)


    #CMYK to GRAY
    gray_img = cv2.cvtColor(CMYK_img, cv2.COLOR_BGR2GRAY)
    # gray_img = cv2.equalizeHist(gray_img)


    #Zack triangular threshold algorithm
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.flatten()
    total = sum(hist)
    max_variance = 0.0
    threshold = 0

    for i in range(256):
        w = sum(hist[:i]) / total
        if w == 0 or w == 1:
            continue
        u = sum([j * hist[j] for j in range(i)]) / sum(hist[:i] + 1e-6)
        v = sum([j * hist[j] for j in range(i, 256)]) / sum(hist[i:] + 1e-6)
        variance = w * (1 - w) * (u - v) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = i
    threshold+=20
    _, binary_image = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)


    #remove shape with smaller area
    #find and draw contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(binary_image)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>1000:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    drawc_image = binary_image.copy()
    drawc_image[mask == 255] = 0
    output_image = binary_image - drawc_image

    #resave the image to enable watershed
    cv2.imwrite("output_image.png", output_image)
    image = cv2.imread("output_image.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #watershed segmentation to separate cell
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2) # Apply morphological operations to remove small noise
    gradient = cv2.subtract(opening, cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)) # Calculate the gradient magnitude
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Apply thresholding to create a binary image
    sure_bg = cv2.dilate(binary, kernel, iterations=3) # Apply morphological operations to remove small holes
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5) # Find sure foreground (unknown region)
    _, sure_fg = cv2.threshold(dist_transform, 0.55*dist_transform.max(), 255, 0)
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
    cv2.imshow("ori",gray_img)
    
    cv2.imshow("bin",binary_image)
    cv2.imshow('Segmented Image', output_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows
    return (number_of_masks)





if __name__ == "__main__":
    number_of_cell = main_function("D:\Image_procesing\Ass\ALL_IDB1\ALL_IDB1\im\Im062_1.jpg")
    print(number_of_cell)








# cv2.imshow("ori",img)
# cv2.imshow("cmyk",CMYK_img)
# cv2.imshow("gray",gray_img)
# cv2.imshow("bin",binary_image)
# cv2.imshow("drawc",output_image)
# print(number_of_masks)
# cv2.imshow('Segmented Image', img)


# cv2.waitKey(0)
# cv2.destroyAllWindows