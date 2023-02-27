import webcolors
from collections import Counter
import numpy as np
import cv2
from detect_circles import detect_circles
import json
import os
from pathlib import Path
from extract_green_plate import extract_green_plate

def get_space(row):
    distances = []
    for i in range(len(row)):
        if i == (len(row)-1):
            break
        distance = row[i+1][0] - row[i][0]
        distances.append(distance)
    min(distances)
    return distance

def check_row(row, space):
    for i in range(len(row)):
        if i == (len(row)-1):
            break
        
        if row[i+1][0]-row[i][0] > (space*1.7):
            new_point = [(row[i][0]+space)*1.05, row[i][1]]
            row.insert(i+1, new_point)
    return row

def Sort_y(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][1] > sub_li[j + 1][1]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li

def Sort_x(sub_li):
    l = len(sub_li)
    for i in range(0, l):
        for j in range(0, l-i-1):
            if (sub_li[j][0] > sub_li[j + 1][0]):
                tempo = sub_li[j]
                sub_li[j]= sub_li[j + 1]
                sub_li[j + 1]= tempo
    return sub_li

def cut_matrix(grid):
    cutted_grids = []
    for row in grid:
        
        if len(row) !=20:
            cutted_row = row[2:-2] 
            
            cutted_grids.append(cutted_row)

    return cutted_grids

def closest(colors,color):
    colors = np.array(colors)
    color = np.array(color)    
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))    
    index_of_smallest = np.where(distances==np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    return smallest_distance 

def get_avarege_color(point, im):
    
    colors = []
    colors_name =[]
    list_of_colors = [[0,0,255],[0,0,0],[255,255,255],[0,255,0],[255,0,0],[255,255,0]]
    for i in range(5):
        for z in range(5):
            color = im[round(point[1]+i), round(point[0]+z)]
            colors.append(color)
            color = im[round(point[1]-i), round(point[0]-z)]
            colors.append(color)
    
    for i in range(len(colors)):         
        closest_color = closest(list_of_colors,colors[i])        
        colors_name.append(webcolors.rgb_to_name((closest_color[0][0],  closest_color[0][1],  closest_color[0][2])))
    data = Counter(colors_name)    
    return data.most_common(1)[0][0]

def get_matrix(image, circles, matrix_Type):
    sort_circles = Sort_y(circles)
    #print("len_sort_Y:", len(sort_circles))
    old_height = sort_circles[0][1]

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
    for row in grids:
        row = Sort_x(row)

    for row in grids:
        space = get_space(row)
        row = check_row(row, space)
    fixed_grids = grids
    if matrix_Type =="image":
        cutted_grids  = cut_matrix(fixed_grids)
    else:
        cutted_grids = fixed_grids

    im = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    if matrix_Type == "image":
        alpha = 1 # Contrast control
        beta = 50 # Brightness control
        im = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        

    color_name_grid = []


    for row in cutted_grids:    
        color_name_row = []
        for point in row:        
            color = get_avarege_color(point, im)        
            color_name_row.append(color)        
        color_name_grid.append(color_name_row)

    return color_name_grid, cutted_grids


def get_similarity(picture_grid, plan_grid):

    row_pic= len(picture_grid)
    column_pic = len(picture_grid[0])

    row_plan = len(plan_grid)
    column_plan = len(plan_grid[0])

    row_diff = row_pic - row_plan +1
    column_diff = column_pic  - column_plan +1

    
    comp_list = []
    

    for row_comp in range(row_diff):
        comp_row_list = []
        for column_comp in range(column_diff):
            number_of_same_colors = 0
            
            
            for i in range(len(plan_grid)):
                
                for z in range(len(plan_grid[i])):
                    plan_color = plan_grid[i][z]
                    pic_color = picture_grid[i+row_comp][z+column_comp]
                    if plan_color == pic_color:
                        number_of_same_colors +=1
            #print(number_of_same_colors)
            
            comp_row_list.append(number_of_same_colors)
            
                
        comp_list.append(comp_row_list)

  
    ################################print("comp_list", comp_list)

    #max_similarity = np.amax(comp_list)
    #index_x = index_2d(comp_list, 90)
    #max_index = np.argmax(comp_list)

    #max_similarity = comp_list[max_index]
    max_similarity, index_x,  index_y = get_max_value(comp_list)
    total_pixel = len(plan_grid) * len(plan_grid[0])
    #print("pos_index",index_x,  index_y) 
    similarity = (max_similarity/total_pixel)*100
    return similarity, (index_x+round(0.5*len(plan_grid))-1), (index_y+round(0.5*len(plan_grid))-1)



def get_max_value(comp_list):
    x = 0
    y = 0
    max_value = 0
    current_x = 0

    for row in comp_list:
        row_max = max(row)
        if row_max > max_value:
            max_value = row_max
            x = current_x
            y = row.index(max_value)
        
        current_x += 1
        
    return max_value, x,  y


def index_2d(list, value):
    for i, x in enumerate(list):
        if value in x:
            return (i, x.index(value))


def safe_new_matrix(template_name, dir_list,  id_list):
    plan_index = 0
    json_data = []
    new_path = "..\..\Templates\\" + template_name

    if not os.path.exists(new_path):
            os.makedirs(new_path)
    else:
        print ("ERROR: File exist")
        return
    
    for dir in dir_list:
        image = cv2.imread(dir)
        circles_template, template_image = detect_circles(image,type_of_image='plan',debug=False)
        matrix_plan_color, matrix_plan_position= get_matrix(image, circles_template, "plan")
        position_matrix_name = "Bauschritt " + str(id_list[plan_index]) + " Positionen"
        
        matrix_plan_position = [[[int(num) for num in point] for point in row] for  row in matrix_plan_position]
        color_matrix_name = "Bauschritt " + str(id_list[plan_index]) + " Farben"
        json_data.append({position_matrix_name : matrix_plan_position, color_matrix_name: matrix_plan_color} )
        plan_index += 1
        cv2.imwrite(new_path + "//" +os.path.basename(dir), image)

    
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
    path = "..\..\Templates\\"
    dir_list = os.listdir(path)
    template_matrix_list = []
    template_name_list = []
    
    
    for dir in dir_list:
        for file in os.listdir(path  +  "//" + dir):
            if ".json" in os.path.basename(file):
                template_specific_matrix =[]
                with open(path  +  "//" + dir + "//" +file, 'r') as openfile:
                    json_object = json.load(openfile)
                template_all_steps_matrix =  []
                template_all_steps_name =[]
                for i in range(len(json_object)):
                    template_step_matrix = []
                    template_step_matrix.append(json_object[i]["Bauschritt " + str(i+1) + " Positionen"])
                    template_step_matrix.append(json_object[i]["Bauschritt " + str(i+1) + " Farben"])
                    template_all_steps_name.append(os.path.splitext(file)[0] + " Bauschritt " + str(i+1))
                    template_all_steps_matrix.append(template_step_matrix)
                template_matrix_list.append(template_all_steps_matrix)
                template_name_list.append(template_all_steps_name)

    return template_matrix_list, template_name_list

def detect_matching_template(image, template_matrix_list, template_name_list):
    rotated_image = extract_green_plate(image, correct_rotation=True, debug=False)
    circles_im,rot_image = detect_circles(rotated_image,type_of_image='photo',debug=False)
    im_image = cv2.cvtColor(rotated_image,cv2.COLOR_BGR2RGB)
    matrix_image, matrix_image_position= get_matrix(im_image, circles_im, "image")

    current_max_similarity = 0
    current_max_index_x = 0 
    current_max_index_y = 0
    current_max_template_index = 0
    current_max_step_index = 0

    template_index = 0
    step_index = 0

    for template in template_matrix_list:
        for step_both_matrixs in template:
            step = step_both_matrixs[1]
            similarity, index_x, index_y= get_similarity(matrix_image,step)
            if similarity > current_max_similarity:
                current_max_similarity = similarity
                current_max_index_x = index_x
                current_max_index_y = index_y
                current_max_template_index = template_index
                current_max_step_index = step_index
            step_index +=1
        template_index +=1

    template_name = template_name_list[current_max_template_index][current_max_step_index]
    image_color_matrix = template_matrix_list[current_max_template_index][current_max_step_index][1]
    template_position_matrix = template_matrix_list[current_max_template_index][current_max_step_index][0]
    return rotated_image, template_name,  matrix_image_position, template_position_matrix, current_max_index_x, current_max_index_y, current_max_similarity

def higlight_target(image, image_position_matrix, template_posotion_matrix, index_x, index_y):
    rest_x = 0
    rest_y = 0

    gab_x= 0
    gab_y= 0
    if (len(template_posotion_matrix)%2) == 0:
        y1 = image_position_matrix[0][0][1]
        y2 = image_position_matrix[-1][0][1]    
        len_y = len(image_position_matrix)-1
        print("y1, y2, len_y",  y1, y2, len_y)
        gab_y = (y2 - y1)//len_y
        rest_y = 0.5 * gab_y

    if len(template_posotion_matrix[0])%2 == 0:
        x1 = image_position_matrix[0][0][0]
        x2 = image_position_matrix[0][-1][0]    
        len_x = len(image_position_matrix[0])-1
        print("x1, x2, len_y",  x1, x2, len_x)
        gab_x = (x2 - x1)//len_x
        rest_x = 0.5 * gab_x

    position = image_position_matrix[index_x][index_y]
    x = int(round(position[0])+rest_x)
    y = int(round(position[1])+rest_y)
    print("x, y:", x,   y)

    template_legnth_y = gab_y * int(round(0.5*len(template_posotion_matrix)))
    template_legnth_x = gab_x * int(round(0.5*len(template_posotion_matrix[0])))
    print("template_legnth_y,template_legnth_x", template_legnth_y,template_legnth_x)

    # start_point_y = int(x - ((x2-x1)/4))
    # start_point_x = int(y - ((y2-y1)/4))

    # end_point_x = int(x + ((x2-x1)/4))
    # end_point_y = int(y + ((y2-y1)/4))

    start_point_y = int(y -  template_legnth_y)
    start_point_x = int(x -   template_legnth_x)
    print("start_point_y,start_point_x", start_point_y,start_point_x)

    end_point_x = int(x +  template_legnth_y)
    end_point_y = int(y +   template_legnth_x)
    print("end_point_y,end_point_x", end_point_y,end_point_x)
    highlighted_image = cv2.rectangle(image, (start_point_x,start_point_y),  (end_point_x,  end_point_y), (0, 0, 0), 10)
    highlighted_image = cv2.circle(image, (x,y), 5, 0, 10)
    return highlighted_image

    

    