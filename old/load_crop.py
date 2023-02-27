from __future__ import annotations
import cv2
import os
import datetime
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from hist import hist_value,hsv_hist_value
from plot_hist import plot_hist, hsv_plot_hist
Current_Date = datetime.datetime.today().strftime ('%d_%b_%Y_%H_%M_%S')

class Building():
    """Base class used to create a comparison by histogramm in order to aid other methods
    of comparing building step. Classes inheriting from this class are ensured to use the same
    order of data points.
    """

    def __init__(self, plan_path: str = "photos/plan/", test_path: str = "photos/test/", scale_fact: float = 1.0, histSize: list = [255,255,255]) -> None:
        self._plan_path = plan_path
        self._test_path = test_path
        self._scale_fact = scale_fact
        self._histSize = histSize
       

    def processing(self, step_num: int = sys.maxsize, load_folder: list[str] = ["all"],debug : bool = False) :
        """Load the images inside the directory (dir_path) and extract histogram from each image by calling functions.
        Afterwards the features are written to hist_results.csv.
        Parameters
        ----------
        step_num : int, optional
            Defines the maximum number of step images/plans to load, by default sys.maxsize (i.e. all of them)
        load_folder : list, optional
            Defines a list of folders from which the images are loaded, by default ["all"]. 
            More than one folder can be loaded by adding the foldername to the list, e.g. load_folder = ['closed_seal_broken', 'closed_sealed']
        debug : bool
            Turns on debug messages for each image
        Returns
        -------
        """
        folder_stop = False
        feature_csv = open('analysis/hist_results.csv', 'w')
        feature_row, comp_list = [],[]
        
        header = ['Image_to_test', 'CompResultStep1', 'CompResultStep2', 'CompResultStep3', 'CompResultStep4', 'CompResultStep5', 'Param']
        write = csv.writer(feature_csv,delimiter=',')   
        write.writerow(header) 
        nb_image = 0
        method = cv2.HISTCMP_BHATTACHARYYA

        for image_path in os.listdir(self._test_path):
            test_img = cv2.imread(self._test_path + image_path) 
            test_img = cv2.resize(test_img,(500,500))
            print('Test Image is:', self._test_path + image_path)
            feature_row.append(image_path)
        hist_test = hsv_hist_value(test_img,ignore_black=True)
        #call plotting function
        #hsv_plot_hist(test_img)
        hsv_plot_hist(test_img)

        for image_path in sorted(os.listdir(self._plan_path)):
            try:
                if True:
                    #print('Image of Step No:',nb_image+1)
                    print('Plan image is:',image_path)
                image = cv2.imread(self._plan_path + image_path)
            
                height, width, _colour_channels = image.shape
                # image = cv2.resize(
                #     image, (int(width*self._scale_fact), int(height*self._scale_fact)), interpolation=cv2.INTER_AREA)
                scaled_height, scaled_width, _colour_channels = image.shape
                image = cv2.resize(image,(500,500))
                # call functions to extract a feature from a single image
                hist_plan = hsv_hist_value(image,ignore_black=True)
                
                hist_compared = cv2.compareHist(hist_plan, hist_test, method)
                #append all features to the row that is to be added
                
                comp_list.append(hist_compared)
                      
            #stopping condition based on the given argument max_num_images   
                if nb_image>=step_num-1:
                    break
                nb_image += 1
            except Exception as e:
                print("Error at image no:",nb_image+1)
                print(e)
                break
        

        #write all extracted features into the csv file for further examination
        #normalize the comparison values to a total of 1 
        print(comp_list)
        normalized = comp_list / np.sum(comp_list)
        for i in range(len(normalized)):
            feature_row.append(normalized[i])
        feature_row.append(str(method))
        write.writerow(feature_row)    
        #close and rename the csv results
        feature_csv.close()
        os.rename(r'analysis/hist_results.csv',r'analysis/hsv_hist_results_' + str(Current_Date) + '.csv')

    def _crop_image(self, image, x: int, y: int, w: int, h: int):
        """Crops an input image to the given parameters.

        Parameters
        ----------
        image : Image
            The image to crop
        x : int
            x-coordinate of the upper-left corner of the new image border
        y : int
            y-coordinate of the upper-left corner of the new image border
        w : int
            width to crop (direction: from left to right)
        h : int
            height to crop (direction: from up to down)

        Returns
        -------
        Image
            Cropped image
        """
        image = image[int(y):int(y+h), int(x):int(x+w)]
        return image

test = Building()
test.processing(step_num=5, debug= False)



