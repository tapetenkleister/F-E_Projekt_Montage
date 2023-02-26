    
import numpy as np
import cv2
from showInMovedWindow import showInMovedWindow

def detect_circles(image:np.ndarray, real_photo:bool, expected_circles_per_longest_side:int, debug:bool=False):
    """Detects circles of the lego bricks or plate and returns a list with their positions and the image .
        Depending of the type of image (photo or plan) the circles are detected. In case of a cropped plan image, the
        number of circles on the longest side need to be known. The bigger the value, the smaller the circles that can be detected.

    Args:
        image (np.array): Image to detect the circles in
        real_photo (bool): Photo by IDS camera(True) or Cropped plan (False)
        expected_circles_per_longest_side(int): How many circles are on the longest side of the plan image.(only necessary for plan images)
        debug (bool, optional): Debug option. Defaults to False.

    Raises:
        NameError: Error if type argument was given wrong

    Returns:
        circle_list:list : List of all circles found with x and y position
        result:np.array : Image with drawn circles
    """ 
    #get parameters of input picture 
    h = image.shape[0]
    w = image.shape[1]
    longest_side_res = max(h,w)  

    #initialize circle list
    circle_list=[]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(12,12))
    clahe = clahe.apply(gray)
    if real_photo:
        #parameterset for detection of circles in photo
        #implies that there are 20x20 circles present
        param1 = 50 #500
        param2 = 16 #200 #smaller value-> more false circles
        minRadius = 8
        minDist = 4*minRadius #mimimal distance from center to center
        maxRadius = 16 #10
    else:
        #parameterset for detection of circles in a cropped plan
        #the amount of expected circles on the longest side is known 
        param1 = 50 #500
        param2 = 12 #200 #smaller value-> more false circles
        minDist = longest_side_res / (expected_circles_per_longest_side + 5.5) #mimimal distance from center to center
        minRadius = int(minDist/4)
        maxRadius = minRadius + 6 #10
    


    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    circles = cv2.HoughCircles(clahe, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    result = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            x, y, r = i[0], i[1], i[2]
            circle_list.append([x, y])

            #x-position, y-position, radius
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            
    else:
        print('No circles found')

    print('Full real plate has 572 circles. (24x24 - 4 edge circles)')
    print('circles found:',len(circles[0]))

    # Show result for testing:
    if debug:
        showInMovedWindow('org', image,0,10)
        showInMovedWindow('gray', gray,310,10)
        showInMovedWindow('clahe', clahe,620,10)
        showInMovedWindow('result', result,930,10)
        cv2.waitKey(0)

    return circle_list,result