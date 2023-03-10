import sys
import time
import cv2


cam = cv2.VideoCapture(0) 

result, image = cam.read()

if result:
    cv2.imshow("Capturing",image)
    key = cv2.waitKey(10)
# image saving
    showPic = cv2.imwrite("git/F-E_Projekt_Montage/Testfoto.jpg",image)
    print(showPic)
    #cv2.waitKey(0)
    cv2.waitKey(500)
    cam.release()
    cv2.destroyWindow("Capturing")
    cv2.destroyAllWindows()

else:
    print("No image detected")
