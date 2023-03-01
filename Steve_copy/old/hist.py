import cv2

def hist_value(image, histSize ):
        hist1 = cv2.calcHist([image], [0,1,2], None,histSize,[0, 256, 0, 256, 0, 256])
        #ignore all black values in the histogramm
        #hist1[0, 0, 0] = 0
        hist1 = cv2.normalize(hist1, hist1).flatten()
        
        return hist1

def hsv_hist_value(image,ignore_black):
        h, s, v = image[:,:,0], image[:,:,1], image[:,:,2]
        hist_h = cv2.calcHist([h],[0],None,[180],[0,180])
        #ignore all black values in the histogramm
        if ignore_black:
                hist_h[0] = 0
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        
        return hist_h