# import the necessary packages

import argparse
import imutils
import time
import cv2
import sys
import numpy as np
from find_closest_corner import find_closest_corner
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str,
                default="DICT_4X4_1000",
                help="type of ArUCo tag to detect")
args = vars(ap.parse_args())
# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}  # verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
        args["type"]))
    sys.exit(0)
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])

arucoParams = cv2.aruco.DetectorParameters()

# Parameter for higher MegaPixel Images https://docs.opencv.org/4.x/d1/dcb/tutorial_aruco_faq.html, 
# https://github.com/opencv/opencv_contrib/issues/2811

arucoParams.adaptiveThreshWinSizeMin = 3  # default 3
arucoParams.adaptiveThreshWinSizeMax = 23  # default 23
arucoParams.adaptiveThreshWinSizeStep = 10  # default 10
arucoParams.adaptiveThreshConstant = 7      # default 7
arucoParams.minMarkerDistanceRate = 0.025  #default 0.05
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

#read the picture
frame = cv2.imread('20230223_124959.jpg')
h, w, c = frame.shape
center = (w/2,h/2)

# scale for intermediate result
scale = 1


frame = imutils.resize(frame, width=int(w*scale), height=int(h*scale))
# detect ArUco markers in the input frame
(corners, ids, rejected) = detector.detectMarkers(frame)
centers = []
inner_corners=[]
print(len(corners))
print(rejected)

# verify *at least* one ArUco marker was detected
if len(corners) == 4:
    # flatten the ArUco IDs list
    ids = ids.flatten()
    # loop over the detected ArUCo corners
    for (markerCorner, markerID) in zip(corners, ids):
        # extract the marker corners (which are always returned
        # in top-left, top-right, bottom-right, and bottom-left
        # order)
        corners = markerCorner.reshape((4, 2))
        
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
        centers.append([cX, cY, markerID])
        

        #compute inner corner
        inner_corner = find_closest_corner(center, corners)
        #print('inner corner',inner_corner)
        inner_corners.append([inner_corner[0],inner_corner[1],markerID])

        # draw the ArUco marker ID on the frame in the top left corner
        cv2.putText(frame, str(markerID),
                    (topLeft[0], topLeft[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
else:
    print('Error: Too many or too less markers detected!')
# show the output frame


output = frame.copy()
output = imutils.resize(output, width=int(w*0.5), height=int(h*0.5))
cv2.imshow("Frame", output)
key = cv2.waitKey(0)



#sort centers by id: from 1 to 4 starting at top left going counterclockwise
centers = sorted(centers, key=lambda x: x[2])
inner_corners = sorted(inner_corners, key=lambda x: x[2])
#print(centers)
#remove the ids from the coordinates (just keep the first to elements)
centers = [x[:2] for x in centers]
inner_corners = [x[:2] for x in inner_corners]

print('centers',centers)
print('inner corners',inner_corners)

#input for the getPerspectiveTransform function
src_pts = np.float32(inner_corners)

# compute the width of the new image
(tl, bl, br, tr) = centers
bottomWidth = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
topWidth = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# compute the width of the new image
rightHeight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
leftHeight = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

# take the maximum of the width and height values to reach
# our final dimensions
maxWidth = max(int(bottomWidth), int(topWidth))
maxHeight = max(int(rightHeight), int(leftHeight))

#alternative height and width calculation which works with 4 markers 1-4 ccw only



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

cv2.imshow("warped_scaled down", warp)


key = cv2.waitKey(0)

cv2.destroyAllWindows()
