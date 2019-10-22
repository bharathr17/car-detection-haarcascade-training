import cv2
import numpy as np
inputimagepath = 'E:/PROJECT ALL/kaggle/project/found2/found-569.jpg'
img = cv2.imread(inputimagepath  )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(thresh, img_bin) = cv2.threshold(gray,  100, 255,cv2.THRESH_BINARY )
%varexp --imshow img_bin
# Invert the image
img_bin = 255-img_bin 
cv2.imwrite("Image_bin.jpg",img_bin)



edges = cv2.Canny(img, 100,200)
%varexp --imshow edges








===============================================================
# Converting the image to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Using the Canny filter to get contours
edges = cv2.Canny(gray, 20, 30)
# Using the Canny filter with different parameters
edges_high_thresh = cv2.Canny(gray, 60, 120)
# Stacking the images to print them together
# For comparison
images = np.hstack((gray, edges, edges_high_thresh))

# Display the resulting frame
cv2.imshow('Frame', canny_images)


===============================================================

# Converting the image to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Smoothing without removing edges.
gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)

# Applying the canny filter
edges = cv2.Canny(gray, 60, 120)
edges_filtered = cv2.Canny(gray_filtered, 60, 120)

# Stacking the images to print them together for comparison
images = np.hstack((gray, edges, edges_filtered))

# Display the resulting frame
cv2.imshow('Frame', canny_images)


===============================================================


fgbg = cv2.createBackgroundSubtractorMOG2(
    history=10,
    varThreshold=2,
    detectShadows=False)

# Read the video
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Converting the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract the foreground
    edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
    foreground = fgbg.apply(edges_foreground)
    
    # Smooth out to get the moving area
    kernel = np.ones((50,50),np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    # Applying static edge extraction
    edges_foreground = cv2.bilateralFilter(gray, 9, 75, 75)
    edges_filtered = cv2.Canny(edges_foreground, 60, 120)

    # Crop off the edges out of the moving area
    cropped = (foreground // 255) * edges_filtered

    # Stacking the images to print them together for comparison
    images = np.hstack((gray, edges_filtered, cropped))


