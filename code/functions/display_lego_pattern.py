import numpy as np
import matplotlib.pyplot as plt

def display_lego_pattern(matrix:np.ndarray)->None:
    """# Define a color map to convert the color strings to RGB values
        #0 = green
        #1 = yellow
        #2 = blue
        #3 = red


    Args:
        matrix (np.ndarray): Array to be displayed
    """ 

    # Get the length of the first row
    first_row_length = len(matrix[0])
    
    # Use slicing to check if all other rows have the same length
    if np.all([len(row) != first_row_length for row in matrix[1:]]):
        print('error')
        raise ValueError("Input matrix doesn't consist of rows with same length")
    

    # color_map = {0: [0, 200, 0], 1: [255, 255, 0],
    #             2: [0, 0, 255], 3: [255, 0, 0],4: [0, 90, 0]}
    color_map = {'green': [0, 200, 0], 'yellow': [255, 255, 0],'white': [255, 255, 255],
                 'blue': [0, 0, 255], 'red': [255, 0, 0],'black': [0, 0, 0],'lime':[0,150,0]}


    # Convert the color matrix to a 3D array of RGB values
    rgb_colors = np.array([[color_map[c] for c in row] for row in matrix])
    return rgb_colors