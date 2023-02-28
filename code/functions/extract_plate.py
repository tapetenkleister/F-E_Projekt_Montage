# import the necessary packages
import imutils
import cv2
import numpy as np
from find_closest_corner import find_closest_corner
from showInMovedWindow import showInMovedWindow
def extract_plate(image:np.ndarray, scale:float=1.0, debug:bool=False):
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
    arucoParams.minMarkerDistanceRate = 0.025  #default 0.05
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

   
    h, w, _ = image.shape
    center = (w/2,h/2)
    image = imutils.resize(image, width=int(w*scale), height=int(h*scale))
    frame = image.copy()
    clean_frame = image.copy()

    # detect ArUco markers in the input frame
    (corners, ids, rejected) = detector.detectMarkers(frame)
    
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

             #compute inner corner
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
        print(f'Error: {len(corners)} detected!')
    # show the output frame
    detected_markers = frame.copy()

    

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

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(bottomWidth), int(topWidth))
    maxHeight = max(int(rightHeight), int(leftHeight))

    # construct destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst_pts = np.array([
        [0, 0],
        [0, maxHeight-1],
        [maxWidth-1, maxHeight-1],
        [maxWidth - 1, 0]], dtype = "float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    warp_clean = cv2.warpPerspective(clean_frame, M, (maxWidth, maxHeight))



    if debug:
        showInMovedWindow('org', image,0,10)
        showInMovedWindow('detected markers', detected_markers,310,10)
        showInMovedWindow('warp', warp,620,10)
        showInMovedWindow('warp clean: result', warp_clean,930,10)
        cv2.waitKey(0)


    return warp_clean