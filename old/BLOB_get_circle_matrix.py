import cv2
import numpy as np

# Load the image
img = cv2.imread('/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/circle_test1.jpg', cv2.IMREAD_GRAYSCALE)

# Set up the blob detector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 20
params.maxThreshold = 255

# Filter by area
params.filterByArea = True
params.minArea = 95
params.maxArea = 15000

# Filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.5

# Filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.1

# Create a blob detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(img)
print('number of detected:')
print(len(keypoints))
# Draw detected blobs as circles
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show the image with detected circles
cv2.imshow("Lego Bricks", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
