import numpy as np
import cv2 
import matplotlib.pyplot as plt
ext = 'jpg'
number_of_images = 5

#size = (400,400)

path = 'code/Pyramide_4rot.'
image = cv2.imread(path+ext)
# Converts the BGR color space of the image to the HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)      

# Threshold of green lego plate  in HSV space H S V
lower = np.array([39, 35, 0])
upper = np.array([100, 255, 170])
# Find green shades inside the image and display them as white in front of a black background
mask = cv2.inRange(hsv, lower, upper)

kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)
#Opening is removing noise from outside a bright contour
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5, iterations=2) 

#Closing is removing noise from inside a bright contour
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_5, iterations=2)

      

# Search the image for contours
contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Get the biggest contour inside the image
biggest_contour = max(contours, key=cv2.contourArea)         

# Create a black canvas and draw all found contours onto it
black_canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
contour_pic = cv2.drawContours(black_canvas.copy(), contours, -1, (0, 0, 255), 2)

# Build a bounding box around the biggest contour found in the image
x_y_w_h = cv2.boundingRect(biggest_contour)
x,y,w,h = x_y_w_h  

# Draw the bounding box
bounding_boxes = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 0, 255), 3) 


# compute rotated rectangle (minimum area)
rect = cv2.minAreaRect(biggest_contour)
angle = rect[-1]
print('angle:',angle)
box = cv2.boxPoints(rect)
box = np.int0(box)
print("bounding box: {}".format(box))
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = box.astype("float32")

# coordinate of the points in box points after the rectangle has been
# straightened
dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")

# the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# directly warp the rotated rectangle to get the straightened rectangle
warped = cv2.warpPerspective(image, M, (width, height))


# # rotate the image to deskew it
# (h, w) = image.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# rotated = cv2.warpAffine(image, M, (w, h),
# 	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



# #Crop the original image to the contour
# roi = image.copy()[y:y+h, x:x+w]








# Show the processing steps of the image
def showInMovedWindow(winname, img, x, y):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.resizeWindow(winname, 300,500)
    cv2.imshow(winname,img)

showInMovedWindow('1mask', mask,0,10)
showInMovedWindow('2opening', opening,310,10)
showInMovedWindow('3closing', closing,620,10)
showInMovedWindow('4bounding_box', bounding_boxes,930,10)
showInMovedWindow('5rotated', warped,1240,10)
cv2.waitKey(0)