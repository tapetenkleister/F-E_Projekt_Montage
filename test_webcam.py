import cv2

cv2.namedWindow("preview")
HIGH_VALUE = 10000
WIDTH = HIGH_VALUE
HEIGHT = HIGH_VALUE
width_set = 640
height_set = 480
capture = cv2.VideoCapture(4)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width_set)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height_set)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width,height)


if capture.isOpened(): # try to get the first frame
    rval, frame = capture.read()
else:
    rval = False

while rval:
    preview = cv2.resize(frame,(width_set,height_set))
    cv2.imshow("preview", preview)
    rval, frame = capture.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

capture.release()
cv2.destroyWindow("preview")