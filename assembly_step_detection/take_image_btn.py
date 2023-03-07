from functions import * 
import cv2

#-----------------------------------------------------------
#this function is called when the button is pressed to take 
#a photo with the IDS camera, it is then saved in the folder
#-----------------------------------------------------------
try:
    #take a photo with the IDS camera
    frame = take_image(exposure_ms=30, debug=False)
    #save the photo in the folder
    cv2.imwrite('Images_Results/image.jpg', frame)
except: 
    print('Error in taking the image')
    