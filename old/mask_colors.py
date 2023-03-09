import numpy as np
import cv2

real = cv2.imread('real.jpg')
plan = cv2.imread('plan.png')
image = cv2.resize(plan, (800, 600))
 
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0,110,0])
upper = np.array([180,255,255])
mask = cv2.inRange(image, lower, upper)

result = cv2.bitwise_and(result, result, mask=mask)




cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()