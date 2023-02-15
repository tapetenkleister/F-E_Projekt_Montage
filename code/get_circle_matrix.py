import numpy as np
import cv2

img = cv2.imread('/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/code/Pyramide_5.jpg')

# define the contrast and brightness value
contrast = 3. # Contrast control ( 0 to 127)
brightness = -60. # Brightness control (0-100)

# call addWeighted function. use beta = 0 to effectively only operate on one image
#adjusted = cv2.addWeighted( img, contrast, img, 0, brightness)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe = clahe.apply(gray)
cv2.imshow('AdaptiveHistEqualizer', clahe)
cv2.waitKey(0)



#blurred = cv2.medianBlur(gray, 5) 
#blurred = cv2.bilateralFilter(clahe,0,sigmaColor=20,sigmaSpace=1)
# edge_detected_image = cv2.Canny(gray, 255/3, 255)



param1 = 50 #500
param2 = 20 #200 #smaller value-> more false circles
minRadius = 8
minDist = 3*minRadius #mimimal distance from center to center
maxRadius = 16 #10

# docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
circles = cv2.HoughCircles(clahe, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
result = img.copy()
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #x-position, y-position, radius
        cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
        print(i)
print('Full plate has 572 circles.')
print('circles found:',len(circles[0]))
# Show result for testing:

# Show the processing steps of the image
def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.resizeWindow(winname, 600,900)
    cv2.imshow(winname,img)

showInMovedWindow('org', img,0,10)
showInMovedWindow('gray', gray,310,10)
showInMovedWindow('clahe', clahe,620,10)
#showInMovedWindow('blurred', blurred,930,10)
showInMovedWindow('5rotated', result,1240,10)
cv2.waitKey(0)