import numpy as np
import cv2 
import matplotlib.pyplot as plt
import time
#Load image with any dimension and print them
ext = 'jpg'
number_of_images = 5
comp_list=[]
size = (400,400)
histSize = [255,255,255]
method = cv2.HISTCMP_INTERSECT
plan = cv2.imread('/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/photos/pi_cam_pyramide/1PI_CAM.jpg')

#plan = cv2.resize(plan,size)
#cv2.imshow("plan",plan)
#tmp = cv2.cvtColor(plan, cv2.COLOR_BGR2GRAY)
#_,alpha = cv2.threshold(tmp,1,255,cv2.THRESH_BINARY)
#--------------------------show the Histogram for each channel-----------------------------------------------
# color = ('r','g','b')
# for i,col in enumerate(color):
#         histr = cv2.calcHist([plan],[i],alpha,[20],[0,256])
#         plt.plot(histr,color = col)
#         plt.xlim([0,20])
# plt.show()

hist1 = cv2.calcHist([plan], [0,1,2], None,histSize,[0, 256, 0, 256, 0, 256])
#ignore all black values in the histogramm
#hist1[0, 0, 0] = 0
hist1 = cv2.normalize(hist1, hist1).flatten()
# plt.title('Input Image')
# plt.plot(hist1)
# plt.xlim([0,256])
# plt.show()

for i in range(1,number_of_images+1):
    path = '/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/photos/pi_cam_pyramide/cropped'+str(i)+'PI_CAM.'
    output_path = '/home/steve/Vorlesungen/FE_Projekt/F-E_Projekt_Montage/photos/pi_cam_pyramide/cropped'+str(i)+'PI_CAM.'
    org_image	= cv2.imread(path+ext)
    #print('Original Dimensions : ',org_image.shape)
    org_image = cv2.resize(org_image,size)
    #convert the image in hsv colorspace for masking
    hsv_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv image",hsv_image)
    blur_img = cv2.GaussianBlur(hsv_image, (7, 7), 0)
    
    # lower boundary green color range values; Hue (0 - 10)
    lower1 = np.array([0, 0, 0])
    upper1 = np.array([30, 255, 255])
    
    # upper boundary green color range values; Hue (160 - 180)
    lower2 = np.array([96,0,0])
    upper2 = np.array([179,255,255])
    
    lower_mask = cv2.inRange(blur_img, lower1, upper1)
    upper_mask = cv2.inRange(blur_img, lower2, upper2)


    full_mask = lower_mask + upper_mask;

    #full_mask = cv2.dilate(full_mask, kernel_3, iterations=3)


    kernel_5 = np.ones((5, 5), np.uint8)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel_5, iterations=6)
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, kernel_5, iterations=6)

    contour = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 

    #get the first position in the contours array sorted by area (biggest contour)
    largest_contour = sorted(contour, key=cv2.contourArea, reverse=True)[0]
    x,y,w,h = cv2.boundingRect(largest_contour)
    roi = org_image.copy()[y:y+h, x:x+w]
    

    # Create a black canvas and draw all found contours onto it
    black_canvas = np.zeros(
        (full_mask.shape[0], full_mask.shape[1], 3), dtype=np.uint8)
    contour_pic = cv2.drawContours(black_canvas, largest_contour, -1, (0, 255, 75), 2) 
    filled = cv2.fillPoly(contour_pic, [largest_contour], (255,255,255))

    #convert to grayscale image with one channel
    filled = cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY)
    prepared= cv2.bitwise_and(org_image,org_image,mask=filled)

    transparent = prepared.copy()
    transparent = cv2.cvtColor(transparent, cv2.COLOR_BGR2BGRA)
    transparent [:,:,3] = filled        

  
    hist2 = cv2.calcHist([org_image], [0,1,2], None,histSize,[0, 256, 0, 256, 0, 256])
    #hist2[0, 0, 0] = 0
    hist2 = cv2.normalize(hist2, hist2).flatten()
    #plt.plot(hist2)
    #plt.xlim([0,256])
    #plt.show()

    comparison = cv2.compareHist(hist1, hist2, method)
    comp_list.append(comparison)
    #print(f'Wert {i} Histogrammvergleich:{comparison}')

    #--------------------------show the Histogram for each channel-----------------------------------------------
    # color = ('r','g','b')
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([org_image],[i],filled,[20],[0,256])
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,20])
    # plt.show()


    #blur the masked image
    #blur_img = cv2.GaussianBlur(full_mask, (3, 3), 0)
    cv2.imshow("blurred hsv image",blur_img)
    cv2.imshow("contour",contour_pic)
    cv2.imshow("end image",prepared)
    cv2.imshow("cropped image",roi)
    cv2.imwrite(output_path + ext,roi)
    cv2.imwrite(output_path + 'png',transparent)
    cv2.waitKey(0)
    plt.close()
    i+=1

print(comp_list)
normalized = comp_list / np.sum(comp_list)

print(normalized)

cv2.destroyAllWindows