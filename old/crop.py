
# import the modules
import os
import cv2
import matplotlib.pyplot as plt
from extract_plate import extract_plate
from detect_circles import detect_circles
from os import listdir
 
# get the path or directory
folder_dir = "/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/photos/ids_bridge"
for i,images in enumerate(os.listdir(folder_dir)):
 
    
    image = cv2.imread(folder_dir+'/'+images)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()
    test = extract_plate(image)
    save=cv2.imwrite(folder_dir+'/crop_bridge'+str(i)+'.jpg',test)
   
   
      
    