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
    
    color_map = {0: [0, 200, 0], 1: [255, 255, 0],
                2: [0, 0, 255], 3: [255, 0, 0],4: [0, 90, 0]}

    # Convert the color matrix to a 3D array of RGB values
    rgb_colors = np.array([[color_map[c] for c in row] for row in matrix])

    # Display the image
    plt.imshow(rgb_colors)
    plt.show()
