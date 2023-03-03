#this file contains all the functions used in the main file
from __future__ import annotations
import numpy as np
import cv2
from showInMovedWindow import showInMovedWindow
import imutils
import math

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
    cv2.moveWindow(winname, x, y)                     # Move it to (x,y)
    cv2.resizeWindow(winname, width,height)           # Resize the window to w x h
    cv2.imshow(winname,img)                           # Show our image inside the created window.

def find_closest_corner(center:tuple, corners:list) -> tuple:
    """Checks cornerpoints distance to a given point and the one with the lowest distance is returned.

    Args:
        center (tuple): Centerpoint with x and y coordinate.
        corners (list): List of corners with x and y coordinates.

    Raises:
        ValueError: Multiple closest corners exist.
        ValueError: No corners provided.

    Returns:
        closest_corner: X and Y coordinates of the corner with lowest distance.
    """    
    # Initialize the minimum distance and closest corner
    min_dist = float('inf')
    closest_corner = None
    
    # Iterate over each corner
    for corner in corners:
        # Compute the Euclidean distance to the center
        dist = math.sqrt((corner[0] - center[0])**2 + (corner[1] - center[1])**2)
        
        # Update the closest corner if this distance is smaller than the current minimum
        if dist < min_dist:
            min_dist = dist
            closest_corner = corner
        elif dist == min_dist:
            # There is a tie between two corners
            raise ValueError("Multiple closest corners exist.")
    
    # Check if a closest corner was found
    if closest_corner is None:
        raise ValueError("No corners provided.")
    
    return closest_corner

def display_lego_pattern(matrix:np.ndarray)->np.ndarray:
    """Displays a matrix of colors as an image

    Args:
        matrix (np.ndarray): Matrix of colors
       
        ValueError: Input matrix doesn't consist of rows with same length

    Returns:
        np.ndarray: Image of the matrix in red, green, blue, yellow
    """   
    # Get the length of the first row
    first_row_length = len(matrix[0])
    
    # Use slicing to check if all other rows have the same length
    if np.all([len(row) != first_row_length for row in matrix[1:]]):
        print('error')
        raise ValueError("Input matrix doesn't consist of rows with same length")
    
    color_map = {'green': [0, 200, 0],'lime': [0, 200, 0], 'yellow': [255, 255, 0],
                 'blue': [0, 0, 255], 'red': [255, 0, 0], 'black': [20, 20, 20]}

    # Convert the color matrix to a 3D array of RGB values
    rgb_colors = np.array([[color_map[c] for c in row] for row in matrix])

    return rgb_colors

def extract_plate(image:np.ndarray, scale:float=1.0, debug:bool=False) ->np.ndarray:
    """This functions returns a perspectively warped image consisting of a plane which is outlined by 4 aruco markers.
    The 4x4_1000 markers are labeled 1-4 and have to be positioned in order top left, bottom left, bottom right and top right.
    This counterclockwise positioning is necessary.


    Args:
        image (np.ndarray): Input image with 4 markers visible.
        scale (float, optional): Downscaling of image if the input is to large. Defaults to 1.0.
        debug (bool, optional):  If TRUE the intermediate steps will be displayed. Defaults to False.


    Returns:
        cropped_image (np.array): Image cropped to the aruco marked plane.
    """
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    arucoParams = cv2.aruco.DetectorParameters()
    # Parameter for higher MegaPixel Images https://docs.opencv.org/4.x/d1/dcb/tutorial_aruco_faq.html, 
    # https://github.com/opencv/opencv_contrib/issues/2811

    arucoParams.adaptiveThreshWinSizeMin = 3  # default 3
    arucoParams.adaptiveThreshWinSizeMax = 23  # default 23
    arucoParams.adaptiveThreshWinSizeStep = 10  # default 10
    arucoParams.adaptiveThreshConstant = 7      # default 7

    #standard of this parameter is to high if the white border of the aruco marker is not wide enough
    arucoParams.minMarkerDistanceRate = 0.025  #default 0.05
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    #assume image is somewhat center
    h, w, _ = image.shape
    center = (w/2,h/2)
    image = imutils.resize(image, width=int(w*scale), height=int(h*scale))
    frame = image.copy()
    clean_frame = image.copy()

    # detect ArUco markers in the input frame
    (corners, ids, _) = detector.detectMarkers(frame)
    
    inner_corners=[]

    # verify exactly 4 ArUco marker were detected
    if len(corners) ==4:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            corners = markerCorner.reshape((4, 2))

             #compute inner corner by finding the closest corner to the center of each marker
            inner_corner = find_closest_corner(center, corners)
            inner_corners.append([inner_corner[0],inner_corner[1],markerID])
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 1)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 1)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 1)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 1)

            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(frame, (cX, cY), 2, (0, 0, 255), -1)
            
            # draw the ArUco marker ID on the frame in the top left corner
            cv2.putText(frame, str(markerID),
                        (topLeft[0], topLeft[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    else:
        #raise Exception("Could not detect 4 Aruco markers")
        raise Exception("Could not detect 4 Aruco markers. Detected: "+str(len(corners))+" markers. Please check the image and try again.")
   

    #sort centers by id: from 1 to 4 starting at top left going counterclockwise
    inner_corners = sorted(inner_corners, key=lambda x: x[2])
    if debug:
        print('Aruco inner corner position and ID:',inner_corners)

    #remove the ids from the coordinates (just keep the first to elements)
    inner_corners = [x[:2] for x in inner_corners]
   
    #input for the getPerspectiveTransform function
    src_pts = np.float32(inner_corners)

    # compute the width of the new image
    (tl, bl, br, tr) = inner_corners
    bottomWidth = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    topWidth = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # compute the width of the new image
    rightHeight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    leftHeight = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach the final dimensions
    maxWidth = max(int(bottomWidth), int(topWidth))
    maxHeight = max(int(rightHeight), int(leftHeight))

    # construct our destination points which will be used to map the screen to a top-down, "birds eye" view
    dst_pts = np.array([
        [0, 0],
        [0, maxHeight-1],
        [maxWidth-1, maxHeight-1],
        [maxWidth - 1, 0]], dtype = "float32")
    
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    detected_markers = frame.copy()

    # directly warp the rotated rectangle to get the straightened rectangle
    warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    warp_clean = cv2.warpPerspective(clean_frame, M, (maxWidth, maxHeight))

    #show results if debug is true
    if debug:
        showInMovedWindow('org', image,0,10)
        showInMovedWindow('detected markers', detected_markers,310,10)
        showInMovedWindow('warp', warp,620,10)
        showInMovedWindow('warp clean: result', warp_clean,930,10)
        cv2.waitKey(0)


    return warp_clean

def get_color_of_roi(point:list, image:np.ndarray, sample_size:int = 12) ->str:
    """This function takes a point (x and y), an image, and samples the average color in a radius of 'radius' pixels around the point.
    It then decides if it's either yellow, red, blue or green and returns this decision.

    Args:
        point (list): list of x and y coordinates
        image (np.ndarray): Image to sample from
        sample_size (int, optional): Square region of interes with length of radius. Defaults to 12.

    Returns:
        string: Name of the sampled color. Either yellow, red, blue or green.
    """
    # Convert the point to integers
    x = int(point[0])
    y = int(point[1])

    # Calculate the coordinates of the region of interest
    x_min = max(x - sample_size, 0)
    y_min = max(y - sample_size, 0)
    x_max = min(x + sample_size, image.shape[1] - 1)
    y_max = min(y + sample_size, image.shape[0] - 1)
    
    # Extract the region of interest
    roi = image[y_min:y_max+1, x_min:x_max+1]
    
    # Calculate the average color of the region of interest
    avg_color = np.mean(roi, axis=(0, 1)).astype(int)
    
    # Define the color thresholds for each color
    yellow_threshold = np.array([0, 250, 250])
    red_threshold = np.array([0, 0, 255])
    blue_threshold = np.array([255, 0, 0])
    green_threshold = np.array([0, 240, 0])
    
    # Calculate the distance between the average color and each color threshold
    distances = [
        np.linalg.norm(avg_color - yellow_threshold),
        np.linalg.norm(avg_color - red_threshold),
        np.linalg.norm(avg_color - blue_threshold),
        np.linalg.norm(avg_color - green_threshold)
    ]
    
    # Decide which color the average color is closest to
    color_decision = np.argmin(distances)
    
    # Return the color decision
    if color_decision == 0:
        return "yellow"
    elif color_decision == 1:
        return "red"
    elif color_decision == 2:
        return "blue"
    elif color_decision == 3:
        return "green"

def detect_circles(image:np.ndarray, real_photo:bool, expected_circles_per_longest_side:int=10, 
                   debug:bool=False) ->tuple[list, np.ndarray]:
    """Detects circles of the lego bricks or plate and returns a list with their positions and the image .
        Depending of the type of image (photo or plan) the circles are detected. In case of a cropped plan image, the
        number of circles on the longest side need to be known. The bigger the value, 
        the smaller the circles that can be detected.

    Args:
        image (np.array): Image to detect the circles in
        real_photo (bool): Photo by IDS camera(True) or Cropped plan (False)
        expected_circles_per_longest_side(int): How many circles are on the longest side of the 
            plan image.(only necessary for plan images)
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
    
    #create a CLAHE object (Arguments are optional) in ord
    #CLAHE means Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(12,12))
    clahe = clahe.apply(gray)

    if real_photo:
        #parameterset for detection of circles in photo
        #implies that there are 20x20 circles present
        param1 = 50 #500
        param2 = 15 #200 #smaller value-> more false circles
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
        raise NameError('No circles found')
    if len(circles[0]) < 350 and real_photo == True:
        #raise name error
        raise NameError('Not enough circles found(less than 350)')
        
   
    print('Circles found:',len(circles[0]))

    # Show result for testing:
    if debug:
        showInMovedWindow('org', image,0,10)
        showInMovedWindow('gray', gray,310,10)
        showInMovedWindow('clahe', clahe,620,10)
        showInMovedWindow('result', result,930,10)
        cv2.waitKey(0)

    return circle_list,result

#function to add padding to an array depending on the size of the array
def add_padding(array_template:np.ndarray or list, array_lego_plate:np.ndarray or list,debug:bool=False) ->np.ndarray:
    """Adds padding to the lego plate array depending on the size of the template array

    Args:
        array_template (np.ndarray):  Array of the template (e.g.10x10 or 4x18)
        array_lego_plate (np.ndarray): Array of the lego plate (20x20)
        debug (bool, optional): Debug option. Defaults to False.
    Returns:
        np.ndarray: Array of the lego plate with padding
    """ 
    #if a list is giben, convert it to an array
    if type(array_template) == list:
        array_template = np.array(array_template)
    if type(array_lego_plate) == list:
        array_lego_plate = np.array(array_lego_plate)   

    #get the size of the template in width and height   
    template_height = array_template.shape[0]
    template_width = array_template.shape[1]

    #size of lego plate is 20x20 and is not to be padded with the string 'black' and different amount top/bottom and left/right
    #padding for left/right sides is int(template_width/2) on each side
    #padding for top/bottom is int(template_height/2) on each side
    #add padding on the left and right side of lego plate
    padding_amount_left_right= int(template_width/2)
    padding_amount_top_bottom = int(template_height/2)

    #debug information
    if debug:
        print('padding_amount_left_right', padding_amount_left_right)
        print('padding_amount_top_bottom', padding_amount_top_bottom)

    #add padding to lego plate
    padded_matrix = np.pad(array_lego_plate, ((padding_amount_top_bottom, padding_amount_top_bottom), (padding_amount_left_right, padding_amount_left_right)),
                           mode='constant', constant_values=('black'))

    return padded_matrix