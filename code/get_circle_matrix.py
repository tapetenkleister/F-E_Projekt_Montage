import numpy as np
import cv2

img = cv2.imread('/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/photos/ids_pyramide/pyramide1.jpg')
h = img.shape[0]
w = img.shape[1]
longest_side_res = max(h,w)

expected_circles_per_longest_side = 24

# call addWeighted function. use beta = 0 to effectively only operate on one image
#adjusted = cv2.addWeighted( img, contrast, img, 0, brightness)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(12,12))
clahe = clahe.apply(gray)
#cv2.imshow('AdaptiveHistEqualizer', clahe)
cv2.waitKey(0)



blurred = cv2.medianBlur(clahe, 3) 
#blurred = cv2.bilateralFilter(clahe,0,sigmaColor=20,sigmaSpace=1)
# edge_detected_image = cv2.Canny(gray, 255/3, 255)

#  #parameterset for detection of circles in photo of ids camera
# param1 = 50 #500
# param2 = 14 #200 #smaller value-> more false circles
# minRadius = 7
# minDist = 4*minRadius #mimimal distance from center to center
# maxRadius = 16 #10

#detection in Plan
param1 = 50 #500
param2 = 12 #200 #smaller value-> more false circles

minDist = longest_side_res / (expected_circles_per_longest_side + 5.5) #mimimal distance from center to center
minRadius = int(minDist/4)
maxRadius = minRadius + 6 #10
print(minDist)
print(minRadius)
print(maxRadius)


# docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
result = img.copy()
if circles is not None:
    #convert the position and radius to integer
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        #x-position, y-position, radius
        cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #print(i)
print('Full plate has 572 (576 lego standard) circles.')
print('circles found:',len(circles[0]))
#print((circles))
# Show result for testing:

# Show the processing steps of the image
def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.resizeWindow(winname, 300,500)
    cv2.imshow(winname,img)

showInMovedWindow('org', img,0,10)
showInMovedWindow('gray', gray,310,10)
showInMovedWindow('clahe', clahe,620,10)
showInMovedWindow('blurred', blurred,930,10)
showInMovedWindow('final', result,1240,10)
cv2.waitKey(0)