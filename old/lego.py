import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

filename="top_view.jpg"
# Read in the image in grayscale
img = cv2.imread('photos/'+filename)
scale_percent = 9 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
blur = cv2.GaussianBlur(hsv, (5,5),0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


values=[90,100,110,120]

edges = cv2.Canny(blur,50,150)
cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
plt.subplot(121),plt.imshow(blur,cmap = 'gray')
plt.title('HSV Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


  #  Standard Hough Line Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

# Determine which openCV version were using
# if cv2.__version__.startswith('2.'):
#     detector = cv2.SimpleBlobDetector()
# else:
#     detector = cv2.SimpleBlobDetector_create()

# # Detect the blobs in the image
# keypoints = detector.detect(resized)
# print(len(keypoints))

# # Draw detected keypoints as red circles
# imgKeyPoints = cv2.drawKeypoints(resized, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)




# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     peri = cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, 0.015 * peri, True)
#     if len(approx) == 4:
#         x,y,w,h = cv2.boundingRect(approx)
#         cv2.rectangle(resized,(x,y),(x+w,y+h),(36,255,12),2)
# cv2.imshow('blur', blur)
# cv2.imshow('thresh', thresh)
# cv2.imshow('image', hsv)

# Display found keypoints
#cv2.imshow("Keypoints", imgKeyPoints)
cv2.waitKey(0)

cv2.destroyAllWindows()