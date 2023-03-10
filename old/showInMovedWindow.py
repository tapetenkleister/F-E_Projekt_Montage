import cv2 
import numpy as np
 # Show the processing steps of the image
def showInMovedWindow(winname:str, img:np.ndarray, x:int=100, y:int=100, width:int=300, height:int=500)->None:  
    """Shows an image as window with configurable position and size.

    Args:
        winname (str): Name
        img (np.ndarray): Image
        x (int, optional): X-Position. Defaults to 100.
        y (int, optional): Y-Position. Defaults to 100.
        width (int, optional): Width. Defaults to 300.
        height (int, optional): Height. Defaults to 500.
    """    
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.resizeWindow(winname, width,height)
    cv2.imshow(winname,img)