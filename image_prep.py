import numpy as np
import cv2 

import time
#Load image with any dimension and print them
path = '/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/photos/pi_cam_pyramide/4PI_CAM.jpg'
org_image	= cv2.imread(path)
#print('Original Dimensions : ',org_image.shape)

#convert the image in hsv colorspace for masking
hsv_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)
#cv2.imshow("hsv image",hsv_image)
blur_img = cv2.GaussianBlur(hsv_image, (9, 9), 0)
cv2.imshow("blurred hsv image",blur_img)
# lower boundary green color range values; Hue (0 - 10)
lower1 = np.array([0, 0, 0])
upper1 = np.array([30, 255, 255])
 
# upper boundary green color range values; Hue (160 - 180)
lower2 = np.array([86,0,0])
upper2 = np.array([179,255,255])
 
lower_mask = cv2.inRange(blur_img, lower1, upper1)
upper_mask = cv2.inRange(blur_img, lower2, upper2)


full_mask = lower_mask + upper_mask;

#full_mask = cv2.dilate(full_mask, kernel_3, iterations=3)


kernel_5 = np.ones((5, 5), np.uint8)
full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel_5, iterations=6)
full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel_5, iterations=6)

contour =cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 

contours = sorted(contour, key=cv2.contourArea, reverse=True)[0]
# Get the biggest contour inside the image
#biggest_contours = max(contour, key=cv2.contourArea)
                  

# Create a black canvas and draw all found contours onto it
black_canvas = np.zeros(
    (full_mask.shape[0], full_mask.shape[1], 3), dtype=np.uint8)
contour_pic = cv2.drawContours(black_canvas, contours, -1, (0, 255, 75), 2) 
filled = cv2.fillPoly(contour_pic, [contours], (255,255,255))
#convert to grayscale image with one channel
filled = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)

prepared= cv2.bitwise_and(org_image,org_image,mask=filled)
        

#blur the masked image
#blur_img = cv2.GaussianBlur(full_mask, (3, 3), 0)
cv2.imshow("contour",contour_pic)
cv2.imshow("end image",prepared)
cv2.waitKey(0)
cv2.destroyAllWindows