import webcolors
from collections import Counter
import numpy as np
import cv2
from detect_circles import detect_circles
import json
import os
from pathlib import Path
from extract_green_plate import extract_green_plate
from extract_plate import extract_plate
from display_lego_pattern import display_lego_pattern
from get_color import get_color_of_roi
#from functions import *

def get_space(row:list):
    """
    Calculates the distance between consecutive points in a row and returns the minimum distance.
    """
    distances = []
    for i in range(len(row)):
        if i == (len(row)-1):
            break
        distance = row[i+1][0] - row[i][0]
        distances.append(distance)
    min(distances)
    return distance

def check_row(row, space, max_len, x_min, x_max):
    """
    Checks a row of points and adds additional points to ensure that the spacing between consecutive points is
    approximately equal to a specified value.

    Arguments:
    row -- list of (x,y) coordinate tuples representing the points in the row
    space -- the desired spacing between consecutive points
    max_len -- the maximum number of points that the row can contain
    x_min -- the minimum x-value that a point in the row can have
    x_max -- the maximum x-value that a point in the row can have
    """
    print("----------------------------------------------- new row ---------------------------------------")
    # Creating empty lists to store points and spaces
    space_list=[]
    point_list=[]
    skip = False
    for i in range(len(row)):
        # Checking if the current point is the first point
        if i ==0:
             # If the point is too far from the left edge of the graph, insert a new point
            if row[i][0] > x_min*1.5:
            # If the point is too far from the right edge of the graph, insert a new point to the right of it

                ##print("insert at i=0")                
                new_point = [(row[i][0]-space), row[i][1]]
                new_row = [new_point]
                new_row.append(row)
                row = new_row
                if len(row) == max_len:
                    print( "break max len")
                    break
                else:
                    continue
        # Checking if the current point is the last point
        if i == (len(row)-1) and len(row)!=max_len:
            ##print("insert at last position", row[i][0])
            x_point = row[i][0]
            if abs(x_max - x_point) > space*0.5:
                new_point = [(x_max), row[i][1]]
                row.insert(i+1, new_point)
                
            if len(row) == max_len:
                    print( "break max len")
                    break
            else:
                # If the row is not at the maximum length, control the spaces and add new points if necessary
                row = control_rows(row, point_list, space_list, max_len)
                
                break
        # Checking if there is a gap between the current point and the next point (insecure spaces)
        if row[i+1][0]-row[i][0] > (space*1) and row[i+1][0]-row[i][0] <= (space*1.6):
            print("add insecure oint at",  i, " position", row[i][0])
            new_point = [(row[i][0]+space), row[i][1]] # calculate the new point to insert
            point_list.append(new_point)
            space_list.append(row[i+1][0]-row[i][0])
            continue

        # Checking if the spacing between the current point and the next point is too large
        if row[i+1][0]-row[i][0] > (space*1.6):
            # Adding a new point between the current point and the next point to reduce the spacing
            
            ##print("insert at",  i, " position", row[i][0])
            new_point = [(row[i][0]+space), row[i][1]]
            row.insert(i+1, new_point)
            skip = True
            # Checking if the row has reached the maximum length
            if len(row) == max_len:
                    print( "break max len")
                    break
            else:
                continue
        if i == range(len(row)):
            print("last control of row")
            row = control_rows(row, point_list, space_list, max_len)
    # Checking if the row has reached the maximum length
    
    if len(row) !=20:
        row = control_rows(row, point_list, space_list, max_len)   


    new_row = Sort_x(row)
    print("new_row", new_row)
    return row

def control_rows(row, point_list, space_list, max_len):
    """
    This function is called if the number of circles on a row does not match the `max_len`.
    It tries to fill the row with missing circles by one of the following ways:
    1- using a list of circles `point_list` if there is the same number of missing circles and circles in the list.
    2- by choosing the largest spaces in the row and placing new circles in the middle of these spaces.
    3- by placing new circles at equal distances in the row to fill the gaps.
    The function takes four arguments:
    - `row`: a list of tuples, each tuple contains the x and y coordinates of a circle center in the row.
    - `point_list`: a list of tuples, each tuple contains the x and y coordinates of a circle center.
    - `space_list`: a list of integers, each integer represents the space between two circles in `row`.
    - `max_len`: the maximum number of circles in `row`.
    The function returns a new list of tuples, each tuple contains the x and y coordinates of a circle center.
    """
    #print("point_list" ,point_list)

    # initialize a new list to store the corrected row
    corrected_row = row
    print("starting control rows")
    print("space_list", space_list)

    # calculate the number of missing circles

    num_missing_circles = max_len - len(row)
    print("num_missing_circles", num_missing_circles)

    # if there are missing circles
    if num_missing_circles !=0:
        # if the number of missing circles is equal to the number of circles in the list, add them directly
        if num_missing_circles == len(point_list):
            print("missing = num insecure")
            for point in range(len(point_list)):
                new_point = point_list[point]
                corrected_row.append(new_point)
                num_missing_circles -=1
        # if the number of missing circles is not equal to the number of circles in the list

        else: 
            print("missing != num insecure")
            for point in range(len(point_list)):
                # find the largest space in the row and place a new circle in the middle of it
                max_space = max(space_list)
                index_max_space = space_list.index(max_space)
                corrected_row.append(point_list[index_max_space])
                del point_list[index_max_space]
                del space_list[index_max_space]
                num_missing_circles -= 1
                # if the row has reached the maximum length, stop adding new circles

                if len(row)==max_len:
                    print( "break max len")
                    break
                # otherwise, continue adding new circles

                else:
                    continue
                
    # if there are still missing circles, place additional circles until the max_length is archieved. (begins with biggest space)
    missing_circles = False
    if num_missing_circles != 0:
        spaces = []
        print("test_row", row)
        for i in range(len(row)-2):
            print("space: ", int(row[i][0]), "-", int(row[i+1][0]))
            spaces.append(abs(int(row[i][0])-int(row[i+1][0])))

        for i in range(num_missing_circles):
            print("spaces", spaces)
            max_space = max(spaces)
            space_index = spaces.index(max_space)
            print("row[space_index][0]", row[space_index][0])
            print("max_space" , max_space)
            print("int(row[space_index][0])+ 0.5*max_space", int(row[space_index][0])+ 0.5*max_space)
            new_point = [int(row[space_index][0])+ 0.5*max_space,row[space_index][1]]
            row.insert(i+1, new_point)
            del spaces[space_index]
            num_missing_circles -= 1
            print(" new_num_missing", num_missing_circles)
            if num_missing_circles ==0:
                missing_circles = False
                break
            if len(spaces) == 0:
                break
        print("filled grid with plan c")

        
            
    return row, missing_circles
            



def Sort_y(sub_li):
    """
    This function sorts a list of points based on their y-coordinates in ascending order.

    Args:
    sub_li: A list of points, where each point is represented as a list [x, y].

    Returns:
    A sorted list of points based on their y-coordinates.
    """
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] > sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li

def Sort_x(sub_li):
    """
    This function sorts a list of points based on their x-coordinates in ascending order.

    Args:
    sub_li: A list of points, where each point is represented as a list [x, y].

    Returns:
    A sorted list of points based on their x-coordinates.
    """

    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][0] > sub_li[j + 1][0]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li


def closest(colors,color):
    """
    This function returns the color in the list 'colors' that is closest to the color 'color'.

    Args:
    colors: A list of colors, where each color is represented as a list [r, g, b].
    color: A color to be compared to the colors in the 'colors' list, represented as a list [r, g, b].

    Returns:
    The color in the 'colors' list that is closest to the 'color' parameter, represented as a list [r, g, b].
    """
    colors = np.array(colors)
    color = np.array(color)    
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))    
    index_of_smallest = np.where(distances==np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    return smallest_distance 

def get_avarege_color(point, im):
    """
    This function returns the name of the most common color of a region of pixels around a point.

    Args:
    point: The coordinates of the center of the region of pixels, represented as a list [x, y].
    im: The image that contains the region of pixels.

    Returns:
    The name of the most common color of the region of pixels around the point.
    """
    # Create empty lists to store the pixel colors and color names
    colors = []
    colors_name =[]
    # Define a list of colors that we want to detect
    list_of_colors = [[0,0,255],[0,255,0],[255,0,0],[255,255,0]]
    
     # Iterate over a 2x2 region of pixels centered around the given point
    for i in range(2):
        for z in range(2):
            # Get the color of the pixel at the given coordinates
            # and append it to the list of colors
            print(round(point[1]+i), round(point[0]+z))
            color = im[round(point[1]+i), round(point[0]+z)]
            colors.append(color)
            print(round(point[1]-i), round(point[0]-z))
            color = im[round(point[1]-i), round(point[0]-z)]
            colors.append(color)
    # Find the name of the closest color for each pixel color in the region

    for i in range(len(colors)):         
        closest_color = closest(list_of_colors,colors[i])        
        colors_name.append(webcolors.rgb_to_name((closest_color[0][0],  closest_color[0][1],  closest_color[0][2])))
     # Use the Counter class to count the occurrences of each color name
    data = Counter(colors_name)    
    return data.most_common(1)[0][0]

def get_matrix(image, circles, matrix_Type):
    """
    A function that generates a matrix of colors from an input image
    based on the locations of circular regions of interest (ROIs)
    represented by the input circles.
    
    Args:
    - image: An input image as a numpy array
    - circles: A list of circles, where each circle is a tuple (x, y, r)
    representing the center coordinates (x, y) and radius (r) of a circular ROI
    - matrix_Type: A string that specifies the type of output matrix: "image"
    for an image-based matrix or "color" for a color-based matrix
    
    Returns:
    - color_name_grid: A 2D list representing the generated matrix of colors
    - cutted_grids: A 2D list representing the generated matrix of circular ROIs
    
    """
    # Sort the circles by y-coordinate
    sort_circles = Sort_y(circles)
    #print("len_sort_Y:", len(sort_circles))
    old_height = sort_circles[0][1]
    # Group circles into rows based on y-coordinate proximity
    grids = []
    rows = []  

    for i in range(len(sort_circles)):
            height = sort_circles[i][1]
            
            if (height-old_height) < (20):
                rows.append(sort_circles[i])
            else:                
                grids.append(rows)
                rows = []
                rows.append(sort_circles[i])
                old_height = height
    grids.append(rows)
    #print("len_grids:", len(grids))
    # Sort circles in each row by x-coordinate
    for row in grids:
        row = Sort_x(row)
    # Find the maximum number of circles in any row
    list_len = [len(row) for row in grids]
    max_len =max(list_len)

    # Find the minimum x-coordinate of any circle
    x_min = 1000000000000

    for row in grids:
        for point in row:
            if point[0] < x_min:
                x_min = point[0]
    # Find the maximum x-coordinate of any circle
    x_max = 0
    for row in grids:
        for point in row:
            if point[0] > x_max:
                x_min = point[0]
    # Check if any row has fewer circles than the maximum number of circles
    # and add additional circles if necessary to create a rectangular grid
    index = 0
    for row in grids:
        space = get_space(row)
        #print("len_row", len(row))
        if len(row)<max_len:
            #print("start cutting")
            #print("row:", index)
            #print("old_row:", row)
            row = check_row(row, space, max_len, x_min, x_max)
            #print("new_len ", len(row))
            #print("new_row:", row)
        index += 1
    fixed_grids = grids
    cutted_grids = fixed_grids
    # Convert input image to RGB color space and adjust contrast/brightness if requested
    im = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    if matrix_Type == "image":
        alpha = 1 # Contrast control
        beta = 50 # Brightness control
        im = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
    # Generate a matrix of colors by averaging the color values within each circle

    color_name_grid = []
    i = 0
    
    index = 0
    for row in grids: #cutted_grids  
       # print("index", index) 
        color_name_row = []
       # print("im_shape:", len(im), len(im[0]))
        for point in row:      
            color = get_avarege_color(point, im)  
            #color = get_color_of_roi(point, im)        
            color_name_row.append(color)        
        color_name_grid.append(color_name_row)
        index += 1
    #print("color_name_grid", color_name_grid)
    for row in color_name_grid:
        print("len color row:", len(row))
    display_lego_pattern(color_name_grid)
    return color_name_grid, cutted_grids


def get_similarity(picture_grid, plan_grid, plan_position_grid):
    """
    Given a picture grid, a plan grid, and a plan position grid, this function finds the best position and orientation
    for the plan grid within the picture grid such that the two grids have the highest possible similarity.

    Args:
        picture_grid (list): A list of lists representing the picture grid. Each element in the list is a color code.
        plan_grid (list): A list of lists representing the plan grid. Each element in the list is a color code.
        plan_position_grid (list): A list of lists representing the position of the plan grid. Each element in the list 
            is a tuple representing the x and y coordinates of the corresponding element in the plan grid.

    Returns:
        A tuple containing the similarity score, the x and y coordinates of the top left corner of the best matching 
        region of the plan grid in the picture grid, the plan position grid after any necessary rotations, and the 
        similarity scores for all positions and rotations.

    """
    # Initialize variables to hold the best matching position and orientation for the plan grid within the picture grid

    

   

    best_comp_list = []
    best_max_similarity = 0
    best_index_x = 0
    best_index_y = 0
    best_rotated_grid = []
    best_rotated_plan_position_grid = []
    rotation = [0, 90, 180, 270]
    # Convert the plan grid and plan position grid to numpy arrays for easier manipulation

    numpy_plan_grid = np.array(plan_grid)
    rotated_plan_grid = numpy_plan_grid

    numpy_plan_position_grid = np.array(plan_position_grid)
    rotated_plan_position_grid = numpy_plan_position_grid
    # Try all possible rotations of the plan grid to find the best matching position and orientation

    for degree in rotation:
        print("degree: ", degree)
        comp_list = []
        
        # Rotate the plan grid and plan position grid by the current degree

        if degree == 0:
            rotated_plan_grid = numpy_plan_grid
            rotated_plan_position_grid = numpy_plan_position_grid
        else:
            rotated_plan_grid = np.rot90(rotated_plan_grid)
            rotated_plan_position_grid = np.rot90(rotated_plan_position_grid)

        padded_matrix = add_padding(rotated_plan_grid, picture_grid,debug=False)
        # Get the dimensions of the rotated plan grid and the difference between the dimensions of the picture grid and
        # the rotated plan grid
        row_plan = len(rotated_plan_grid)
        column_plan = len(rotated_plan_grid[0])

        row_pic= len(padded_matrix)
        column_pic = len(padded_matrix[0])

        row_diff = row_pic - row_plan +1
        column_diff = column_pic  - column_plan +1
        # Iterate over all possible positions of the rotated plan grid within the picture grid

        for row_comp in range(row_diff):
            comp_row_list = []
            for column_comp in range(column_diff):
                number_of_same_colors = 0
                # Calculate the similarity score for the current position by comparing the color codes of each element in
                # the plan grid and the corresponding element in the picture grid
                
                for i in range(len(rotated_plan_grid)):
                    
                    for z in range(len(rotated_plan_grid[i])):
                        plan_color = rotated_plan_grid[i][z]
                        pic_color = padded_matrix[i+row_comp][z+column_comp]
                        if plan_color == pic_color:
                            number_of_same_colors +=1
                #print(number_of_same_colors)
                
                comp_row_list.append(number_of_same_colors)
                
                    
            comp_list.append(comp_row_list)
        # Get the maximum similarity
        max_similarity, index_x,  index_y = get_max_value(comp_list)
        if max_similarity >= best_max_similarity:
            print("higher sim found")
            best_max_similarity = max_similarity
            best_index_x = index_x
            best_index_y = index_y
            best_comp_list = comp_list
            best_rotated_grid = rotated_plan_grid
            best_rotated_plan_position_grid = rotated_plan_position_grid

    print("best_rot_grid", best_rotated_grid)
    print("best_max_similarity", best_max_similarity)
    total_pixel = len(best_rotated_grid) * len(best_rotated_grid[0])
    print("best index x y", best_index_x, best_index_y)
    print("index x y in image", best_index_x+round(0.5*len(best_rotated_grid[0]))-1,best_index_y+round(0.5*len(best_rotated_grid))-1 )
    #print("pos_index",index_x,  index_y) 
    similarity = (best_max_similarity/total_pixel)*100
    im_index_x = index_x-1
    im_index_y = index_y-1
    # im_index_x = best_index_x+round(0.5*len(best_rotated_grid[0]))-1
    # im_index_y = best_index_y+round(0.5*len(best_rotated_grid))-1
    return similarity, im_index_x, im_index_y, best_rotated_plan_position_grid, best_comp_list



def get_max_value(comp_list):
    """
    Given a comparison list, find the maximum value and its corresponding indices.

    Parameters:
    comp_list (list of lists): The comparison list to search through.

    Returns:
    max_value (float or int): The maximum value found in the list.
    x (int): The row index where the maximum value was found.
    y (int): The column index where the maximum value was found.
    """
    x = 0
    y = 0
    max_value = 0
    current_x = 0
# Iterate through each row in the comparison list.
    for row in comp_list:
        row_max = max(row)
        # If the maximum value in the current row is greater than the maximum value found so far,
        # update the maximum value and the indices where it was found.
        if row_max > max_value:
            max_value = row_max
            x = current_x
            y = row.index(max_value)
        
        current_x += 1# Increment the current row index.
        
    return max_value, x,  y



def safe_new_matrix(template_name, dir_list,  id_list, longest_side):
    """
    This function creates a new directory with the given template name and saves the position and color matrices of 
    each image in the directory list as JSON files in the newly created directory. 

    Args:
    - template_name (str): The name of the template to be created.
    - dir_list (list of str): A list of image file paths to be processed.
    - id_list (list of int): A list of ids to be assigned to each image in the JSON data.
    - longest_side (int): The expected number of circles to be detected along the longest side of the image.

    Returns:
    - None

    Raises:
    - None
    """

    # Initialize variables and create the new directory
    plan_index = 0
    json_data = []
    new_path = "..\..\Templates\\" + template_name

    if not os.path.exists(new_path):
            os.makedirs(new_path)
    else:
        print ("ERROR: File exist")
        return
    
    for dir in dir_list:
        # Read the image and detect the circles
        image = cv2.imread(dir)
        circles_template, template_image = detect_circles(image,real_photo=False,expected_circles_per_longest_side=longest_side, debug=False)
        # Extract the position and color matrices from the image
        matrix_plan_color, matrix_plan_position= get_matrix(image, circles_template, "plan")
        position_matrix_name = "Bauschritt " + str(id_list[plan_index]) + " Positionen"
        
        matrix_plan_position = [[[int(num) for num in point] for point in row] for  row in matrix_plan_position]
        color_matrix_name = "Bauschritt " + str(id_list[plan_index]) + " Farben"
        # Store the matrices and their associated ids in a dictionary and append it to the JSON data list

        json_data.append({position_matrix_name : matrix_plan_position, color_matrix_name: matrix_plan_color} )
        plan_index += 1
        # Save the image to the new directory
        cv2.imwrite(new_path + "//" +os.path.basename(dir), image)

    # Save the JSON data to a file in the new directory
    if os.path.exists(new_path):
        new_template_file = new_path + "\\" + template_name + ".json"    
        if os.path.isfile(new_template_file):
            print ("ERROR: File exist")
        else:
            with open(new_template_file, 'w') as f:
                json.dump(json_data, f)
    else:
        print ("ERROR: Dir not exist")
        return
    
def open_saved_matrix():
    """
    This function opens the saved matrix from the file system and returns it.

    Returns:
    - template_matrix_list: a list of matrices containing plan positions and colors for each step of each template. 
    - template_name_list: a list of strings containing names of each step of each template.
    """
    path = "..\..\Templates\\"
    dir_list = os.listdir(path)
    template_matrix_list = []
    template_name_list = []
    
    # Iterate over directories in the template path
    for dir in dir_list:
        # Check if the file is a json file
        for file in os.listdir(path  +  "//" + dir):
            if ".json" in os.path.basename(file):
                template_specific_matrix =[]
                with open(path  +  "//" + dir + "//" +file, 'r') as openfile:
                    json_object = json.load(openfile)
                template_all_steps_matrix =  []
                template_all_steps_name =[]
                # Iterate over each step of the template
                for i in range(len(json_object)):
                    template_step_matrix = []
                    # Add the matrix of plan positions and colors for each step to a list

                    template_step_matrix.append(json_object[i]["Bauschritt " + str(i+1) + " Positionen"])
                    template_step_matrix.append(json_object[i]["Bauschritt " + str(i+1) + " Farben"])
                    # Append the name of each step to a list
                    template_all_steps_name.append(os.path.splitext(file)[0] + " Bauschritt " + str(i+1))
                    template_all_steps_matrix.append(template_step_matrix)
                # Append the list of matrices of each template to a list
                template_matrix_list.append(template_all_steps_matrix)
                # Append the list of names of each step of each template to a list

                template_name_list.append(template_all_steps_name)

    return template_matrix_list, template_name_list

def detect_matching_template(image, template_matrix_list, template_name_list):
    """
    Detects a matching template from the provided list of template matrices for a given input image.

    Parameters:
    image (numpy.ndarray): The input image as a NumPy array.
    template_matrix_list (list): The list of template matrices.
    template_name_list (list): The list of names of the templates.

    Returns:
    numpy.ndarray: The rotated input image.
    str: The name of the template that matches the input image.
    list: The position of each cell in the matrix of the input image.
    list: The position of each cell in the matched template.
    int: The x-coordinate of the matched template's top-left corner in the input image.
    int: The y-coordinate of the matched template's top-left corner in the input image.
    float: The similarity between the input image and the matched template.
    list: A list of dictionaries containing comparison data between the input image and the matched template.
    """
    # Extract green plate from input image and correct its rotation

    rotated_image = extract_green_plate(image, correct_rotation=True, debug=False)
    # Detect circles in the rotated image
    circles_im,rot_image = detect_circles(rotated_image,real_photo=True,expected_circles_per_longest_side=10,debug=False)
    # Convert rotated image to RGB
    im_image = cv2.cvtColor(rotated_image,cv2.COLOR_BGR2RGB)
    # Extract matrix from the image using detected circles
    matrix_image, matrix_image_position= get_matrix(im_image, circles_im, "image")
    # Initialize variables to keep track of best match
    current_max_similarity = 0
    current_max_index_x = 0 
    current_max_index_y = 0
    current_max_template_index = 0
    current_max_step_index = 0
    current_plan_position_grid = []

    template_index = 0
    step_index = 0
    # Loop over all templates and their steps
    for template in template_matrix_list:
        step_index = 0
        for step_both_matrixs in template:
            # Extract position and color matrices for the current step
            step_position_matrix = step_both_matrixs[1]
            step_color_matrix = step_both_matrixs[0]
            # Compute similarity between the image matrix and the current template step
            similarity, index_x, index_y, rotated_plan_position_grid, comp_list= get_similarity(matrix_image,step_position_matrix, step_color_matrix)
            # Update the best match if the current similarity is higher than the previous max
            if similarity > current_max_similarity:
                current_max_similarity = similarity
                current_max_index_x = index_x
                current_max_index_y = index_y
                current_max_template_index = template_index
                current_max_step_index = step_index
                current_plan_position_grid = rotated_plan_position_grid
            step_index +=1
        template_index +=1
    # Print some debug information
    print("end templ pos x, y ", current_max_index_x, current_max_index_y)
    template_name = template_name_list[current_max_template_index][current_max_step_index]
    print("template_name", template_name)
    print("sim", current_max_similarity)
    return rotated_image, template_name,  matrix_image_position, current_plan_position_grid, current_max_index_x, current_max_index_y, current_max_similarity, comp_list

def higlight_target(image, image_position_matrix, template_posotion_matrix, index_x, index_y):
    """
    Given an input image, highlight the area of the target specified by its position in the image matrix.

    Args:
    - image: the input image
    - image_position_matrix: the matrix of positions of the image pixels
    - template_position_matrix: the matrix of positions of the target pixels
    - index_x: the x-index of the target in the image matrix
    - index_y: the y-index of the target in the image matrix

    Returns:
    - highlighted_image: the input image with the target area highlighted

    """
    print("template_posotion_matrix", len(template_posotion_matrix[0]), len(template_posotion_matrix))
    # Initialize variables to compute the gaps and rests between the target and the image pixels

    rest_x = 0
    rest_y = 0
    # Compute the gap and rest for the y-dimension (center of two circles)
    gab_x= 0
    gab_y= 0
    if (len(template_posotion_matrix)%2) == 0:
        y1 = image_position_matrix[0][0][1]
        y2 = image_position_matrix[-1][0][1]    
        len_y = len(image_position_matrix)-1
        print("y1, y2, len_y",  y1, y2, len_y)
        gab_y = (y2 - y1)//len_y
        rest_y = 0.5 * gab_y
    # Compute the gap and rest for the x-dimension
    if len(template_posotion_matrix[0])%2 == 0:
        x1 = image_position_matrix[0][0][0]
        x2 = image_position_matrix[0][-1][0]    
        len_x = len(image_position_matrix[0])-1
        print("x1, x2, len_y",  x1, x2, len_x)
        gab_x = (x2 - x1)//len_x
        rest_x = 0.5 * gab_x
    print("index x, y", index_x, index_y)
    # Compute the pixel position of the target
    position = image_position_matrix[index_x][index_y]
    i = 0
    for row in image_position_matrix:
        print("line", i, row)
        i +=1
    print("position" , position)
    x = int(round(position[0])+rest_x)
    y = int(round(position[1])+rest_y)
    print("x, y:", x,   y)
     # Compute the length of the target in the y- and x-dimensions
    template_legnth_y = gab_y * int(round(0.5*len(template_posotion_matrix)))
    template_legnth_x = gab_x * int(round(0.5*len(template_posotion_matrix[0])))
    print("template_legnth_y,template_legnth_x", template_legnth_y,template_legnth_x)
# Compute the start and end points of the target area in the image
    start_point_y = int(y -  template_legnth_y)
    start_point_x = int(x -   template_legnth_x)
    print("start_point_y,start_point_x", start_point_y,start_point_x)

    end_point_x = int(x +  template_legnth_x)
    end_point_y = int(y +   template_legnth_y)
     # Highlight the target area in the image with a rectangle and a circle
    print("end_point_y,end_point_x", end_point_y,end_point_x)
    highlighted_image = cv2.rectangle(image, (start_point_x,start_point_y),  (end_point_x,  end_point_y), (0, 0, 0), 10)
    highlighted_image = cv2.circle(image, (x,y), 5, 0, 10)
    return highlighted_image

    

def add_padding(array_template:np.ndarray or list, array_lego_plate:np.ndarray or list,debug:bool=False) ->np.ndarray:
    """Adds padding to the lego plate array depending on the size of the template array

    Args:
        array_template (np.ndarray):  Array of the template (e.g.10x10 or 4x18)
        array_lego_plate (np.ndarray): Array of the lego plate (20x20)
        debug (bool, optional): Debug option. Defaults to False.
    Returns:
        np.ndarray: Array of the lego plate with padding
    """ 
    #if a list is giben, convert it to an array
    if type(array_template) == list:
        array_template = np.array(array_template)
    if type(array_lego_plate) == list:
        array_lego_plate = np.array(array_lego_plate)   

    #get the size of the template in width and height   
    template_height = array_template.shape[0]
    template_width = array_template.shape[1]

    #size of lego plate is 20x20 and is not to be padded with the string 'black' and different amount top/bottom and left/right
    #padding for left/right sides is int(template_width/2) on each side
    #padding for top/bottom is int(template_height/2) on each side
    #add padding on the left and right side of lego plate
    padding_amount_left_right= int(template_width/2)
    padding_amount_top_bottom = int(template_height/2)

    #debug information
    if debug:
        print('padding_amount_left_right', padding_amount_left_right)
        print('padding_amount_top_bottom', padding_amount_top_bottom)

    #add padding to lego plate
    padded_matrix = np.pad(array_lego_plate, ((padding_amount_top_bottom, padding_amount_top_bottom), (padding_amount_left_right, padding_amount_left_right)),
                           mode='constant', constant_values=('black'))

    return padded_matrix