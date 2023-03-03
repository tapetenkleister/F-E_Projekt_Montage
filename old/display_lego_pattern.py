import numpy as np


def display_lego_pattern(matrix:np.ndarray)->np.ndarray:
    """Displays a matrix of colors as an image

    Args:
        matrix (np.ndarray): Matrix of colors
       
        ValueError: Input matrix doesn't consist of rows with same length

    Returns:
        np.ndarray: Image of the matrix in red, green, blue, yellow
    """   
    # Get the length of the first row
    first_row_length = len(matrix[0])
    
    # Use slicing to check if all other rows have the same length
    if np.all([len(row) != first_row_length for row in matrix[1:]]):
        print('error')
        raise ValueError("Input matrix doesn't consist of rows with same length")
    
    color_map = {'green': [0, 200, 0],'lime': [0, 200, 0], 'yellow': [255, 255, 0],
                 'blue': [0, 0, 255], 'red': [255, 0, 0]}

    # Convert the color matrix to a 3D array of RGB values
    rgb_colors = np.array([[color_map[c] for c in row] for row in matrix])

    return rgb_colors