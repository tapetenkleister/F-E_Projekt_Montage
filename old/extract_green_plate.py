import numpy as np
import cv2 
from showInMovedWindow import showInMovedWindow

def extract_green_plate (image:np.ndarray, correct_rotation:bool= False, debug:bool=False):
    """This function crops out the green lego plate from an image. Optionally it also rotates the plate to be parallel to the screen.


    Args:
        image (np.array): Image to be cropped into
        correct_rotation (bool, optional): If TRUE the output image will contain a correctly rotated plate. Defaults to False.
        debug (bool, optional): If TRUE the intermediate steps will be displayed. Defaults to False.

    Returns:
        cropped_image (np.array): Image cropped to the green plate and rotated if option is on.
    """    
    
    # Converts the BGR color space of the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)      

    # Threshold of green lego plate  in HSV space H S V
    lower = np.array([39, 35, 0])
    upper = np.array([100, 255, 170])

    # Find green shades inside the image and display them as white in front of a black background
    mask = cv2.inRange(hsv, lower, upper)

    kernel_5 = np.ones((5, 5), np.uint8)

    #Opening is removing noise from outside a bright contour
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5, iterations=2) 

    #Closing is removing noise from inside a bright contour
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_5, iterations=2)

    # Search the image for contours
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Get the biggest contour inside the image by area
    biggest_contour = max(contours, key=cv2.contourArea)         

    # Create a black canvas and draw all found contours onto it
    black_canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    contour_pic = cv2.drawContours(black_canvas.copy(), contours, -1, (0, 0, 255), 2)

    # Build a bounding box around the biggest contour found in the image
    x_y_w_h = cv2.boundingRect(biggest_contour)
    x,y,w,h = x_y_w_h  

    # Draw the bounding box
    bounding_boxes = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 0, 255), 3) 

    #If rotation correction is turned on compute the angle between a rotated rectangle to horizontal
    if correct_rotation:
        # create rotated rectangle (minimum area)
        rect = cv2.minAreaRect(biggest_contour)
        angle = rect[-1]
        

        #Get coordinates of the rotated bounding box
        rotated_box = cv2.boxPoints(rect)
        rotated_box = np.int0(rotated_box)
        

        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = rotated_box.astype("float32")

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
        cropped_image = warped
    else: 
        cropped_image = image[int(y):int(y+h), int(x):int(x+w)]
        
    if debug:
        print('angle:',angle)
        print("bounding box: {}".format(rotated_box))
        showInMovedWindow('1mask', mask,0,10)
        showInMovedWindow('2opening', opening,310,10)
        showInMovedWindow('3closing', closing,620,10)
        showInMovedWindow('4contour',contour_pic,930,300)
        showInMovedWindow('5bounding_box', bounding_boxes,930,10)
        showInMovedWindow('6cropped', cropped_image,1240,10)
        cv2.waitKey(0)

    return cropped_image


