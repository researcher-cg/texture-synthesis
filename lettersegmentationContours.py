import cv2

# Read the image
image = cv2.imread('Images/1/corrupt/tex_ruins1_scratched.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocessing (grayscale conversion and thresholding)
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Connected Component Analysis (CCA)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
min_area_threshold = 20

# Iterate through connected components
for i in range(1, num_labels):
    # Filter out small components
    if stats[i, cv2.CC_STAT_AREA] < min_area_threshold:
        continue
    
    # Get bounding box coordinates
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    
    # Draw bounding box on original image (optional)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # Extract letter ROI from original image
    letter_roi = image[y:y+h, x:x+w]
    
    # Perform further processing or recognition on the letter ROI
    
# Display the image with bounding boxes (optional)
cv2.imshow('Segmented Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()