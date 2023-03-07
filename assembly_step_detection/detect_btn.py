from functions import *
from matrix_to_be_deleted import*
import cv2
import  sys
#-----------------------------------------------------------
#this function is called when the button is pressed to detect the assembly step
#-----------------------------------------------------------

try:
    #load the image taken with IDS camera if it is available otherwise print an error message
    img = cv2.imread('Images_Results/image.jpg')
    if img is None:
        print('No image available')
        sys.exit()
    else:
        #extract the lego plate
        extracted_lego_plate = extract_plate(img, scale=1, debug=False)

except:
    print('Error in extracting the lego plate')
    

try:
    #detect circles in the extracted lego plate
    detected_circles_list,detected_circles_image = detect_circles(extracted_lego_plate, real_photo=True, debug=False)
    cv2.imwrite('Images_Results/circles.jpg', detected_circles_image)
except:
    print('Error in detecting circles')

try:
    #read in all available templates of steps
    template_matrix_list, template_name_list = open_saved_matrix('Templates/')

    #compare all templates by folding their color_matrix over the color_matrix of the image (rotation included)
    color_matrix, detected_assembly_step,  matrix_image_position, template_position_matrix, index_x, index_y, max_similarity, comp_list = detect_matching_template(detected_circles_list, template_matrix_list, template_name_list)
    cv2.imwrite('Images_Results/color_matrix.jpg',color_matrix)

    # write detected_assembly_step, position and rotation to a txt.file for examination
    with open('Images_Results/result.txt', 'w') as f:
        f.write(f'Detected assembly step: ||' + detected_assembly_step + '||')
        f.write(f' with '+str(max_similarity)+'%'+' similarity\n')
        f.write(f'Center position in x and y: '+str(matrix_image_position)+'\n')
        f.write(f'Rotation: '+str(template_position_matrix)+'°'+'\n')

except:
    print('Error in detecting assembly step')

try:
    #generate an image with the detected step in green frame
    result_image = higlight_target(extracted_lego_plate, matrix_image_position, template_position_matrix, index_x, index_y)
    cv2.imwrite('Images_Results/result.jpg', result_image)
except:
    print('Error in highlighting the final result')
