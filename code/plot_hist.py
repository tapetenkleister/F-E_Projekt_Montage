from __future__ import annotations
import cv2
import os
import datetime
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(img):
    color = ('b','g','r')
    for channel,col in enumerate(color):
        histr = cv2.calcHist([img],[channel],None,[256],[0,256])
        #ignore all black values in the histogramm
        #histr[0] = 0
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title('Histogram for color scale picture')
    plt.show()

    # while True:
    #     k = cv2.waitKey(0) & 0xFF     
    #     if k == 27: break             # ESC key to exit 
    # cv2.destroyAllWindows()
def hsv_plot_hist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = img[:,:,0], img[:,:,1], img[:,:,2]
    hist_h = cv2.calcHist([h],[0],None,[180],[0,180])
    hist_h[0] = 0
    #hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    #hist_v = cv2.calcHist([v],[0],None,[256],[0,256])
    
    plt.hist(hist_h, color='r', label="h")

    #plt.plot(hist_h, color='r', label="h")
    
    #plt.plot(hist_s, color='g', label="s")
    #plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.show()
    # while True:
    #     k = cv2.waitKey(0) & 0xFF     
    #     if k == 27: break             # ESC key to exit 
    # cv2.destroyAllWindows()