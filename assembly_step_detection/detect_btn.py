from functions import *
import cv2 
import time
import sys
import datetime

#-----------------------------------------------------------
#this function is called when the button is pressed to detect the assembly step
#-----------------------------------------------------------
#start the clock to measure the time needed for the detection
start = time.time()
try:
    #load the image taken with IDS camera if it is available otherwise print an error message
    img = cv2.imread('Images_Results/image.jpg')

        #extract the lego plate with aruco markers
    extracted_lego_plate = extract_plate(img, scale=1, debug=False)
except Exception as e:
    #in case no plate can be extracted overwrite the circles.jpg, color_matrix.png and result.jpg with a black image
    cv2.imwrite('Images_Results/circles.jpg', np.zeros((100,100,3), np.uint8))
    cv2.imwrite('Images_Results/color_matrix.png', np.zeros((100,100,3), np.uint8))
    cv2.imwrite('Images_Results/result.jpg', np.zeros((100,100,3), np.uint8))
    print('Error in extracting lego plate')
    print(e)
    with open('Images_Results/result.txt', 'w') as f:
        f.write(f'Something went wrong in extracting the plate. Make sure that 4 aruco markers are sharp and visible.\n')
        f.write(f'Error: '+str(e)+'\n')
        sys.exit()

try:
    #detect circles in the extracted lego plate
    detected_circles_list,detected_circles_image = detect_circles(extracted_lego_plate, real_photo=True, debug=False)
    cv2.imwrite('Images_Results/circles.jpg', detected_circles_image)
except Exception as e:
    #in case no circles can be extracted overwrite the circles.jpg, color_matrix.png and result.jpg with a black image
    cv2.imwrite('Images_Results/circles.jpg', np.zeros((100,100,3), np.uint8))
    cv2.imwrite('Images_Results/color_matrix.png', np.zeros((100,100,3), np.uint8))
    cv2.imwrite('Images_Results/result.jpg', np.zeros((100,100,3), np.uint8))
    print('Error in detecting circles')
    print(e)
    with open('Images_Results/result.txt', 'w') as f:
        f.write(f'Something went wrong in detecting circles on the extracted plate.\n')
        f.write(f'Error: '+str(e)+'\n')
        sys.exit()

try:
    #read in all available templates of steps
    template_matrix_list, template_name_list = open_saved_matrix('Templates/')

    #compare all templates by folding their color_matrix over the color_matrix of the image (rotation included)
    color_matrix, detected_assembly_step,  matrix_image_position, template_position_matrix, index_x, index_y, max_similarity, comp_list, rotation_with_best_similarity = detect_matching_template(extracted_lego_plate, detected_circles_list, template_matrix_list, template_name_list)
    color_matrix = color_matrix.astype(np.uint8)

    #convert the color matrix to RGB for saving
    color_matrix = cv2.cvtColor(color_matrix, cv2.COLOR_RGB2BGR)
    cv2.imwrite('Images_Results/color_matrix.png',color_matrix)
except Exception as e:
    #in case no color matrix can be extracted overwrite color_matrix.png and result.jpg with a black image
    cv2.imwrite('Images_Results/color_matrix.png', np.zeros((100,100,3), np.uint8))
    cv2.imwrite('Images_Results/result.jpg', np.zeros((100,100,3), np.uint8))
    print('Error in detecting matching template')
    print(e)
    with open('Images_Results/result.txt', 'w') as f:
        f.write(f'Something went wrong in detecting a matching template step.\n')
        f.write(f'Error: '+str(e)+'\n')
        sys.exit()

try:
    #generate an image with the detected step in green frame
    result_image = higlight_target(extracted_lego_plate, matrix_image_position, template_position_matrix, index_x, index_y)
    cv2.imwrite('Images_Results/result.jpg', result_image)
except Exception as e:
    #in case no result image can be extracted overwrite result.jpg with a black image
    cv2.imwrite('Images_Results/result.jpg', np.zeros((100,100,3), np.uint8))
    print('Error in highlighting the final result')
    print(e)
    with open('Images_Results/result.txt', 'w') as f:
        f.write(f'Something went wrong in highlighting the building in the result image.\n')
        f.write(f'Error: '+str(e)+'\n')
        sys.exit()

#clock the time needed for the detection
end = time.time()
duration = end - start

# write detected_assembly_step, position and rotation to a txt.file for examination
with open('Images_Results/result.txt', 'w') as f:
    f.write(f'Detected assembly step: ' + detected_assembly_step+'\n')
    f.write(f'Similarity: '+str(max_similarity)+'% of proposed step was found\n')
    f.write(f'Center position in x and y: '+str(index_y+1)+' , '+str(index_x+1)+'\n')
    f.write(f'Rotation: '+str(rotation_with_best_similarity)+'°'+'\n')
    f.write(f'Time needed for detection: '+str(round(duration*1000,2))+'ms'+'\n')

# txt file for examination and extended every time the button is pressed
with open('Images_Results/history.txt', 'a') as f:
    f.write(f'Detected assembly step: ' + detected_assembly_step+'\n')
    f.write(f'Similarity: '+str(max_similarity)+'% of proposed step was found\n')
    f.write(f'Center position in x and y: '+str(index_y+1)+' , '+str(index_x+1)+'\n')
    f.write(f'Rotation: '+str(rotation_with_best_similarity)+'°'+'\n')
    f.write(f'Time needed for detection: '+str(round(duration*1000,2))+'ms'+'\n')
    f.write(f'time: '+str(datetime.datetime.now())+'\n')
