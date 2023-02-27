import cv2
import numpy as np

   
def nothing(x):
  pass

cv2.namedWindow('Image')
img = cv2.imread("real2.jpg")

low = 1
high = 100

cv2.createTrackbar('Blur', 'Image',low,high,nothing)
while (True):
    ksize = cv2.getTrackbarPos('Blur', 'Image')
    if ksize%2==0:
        ksize+=1
    image = cv2.GaussianBlur(img,(ksize,ksize),0)
    
    cv2.imshow('Image', image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()