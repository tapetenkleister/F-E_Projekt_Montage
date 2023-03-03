import numpy as np

def get_color_of_roi(point:list, image:np.ndarray, sample_size:int = 12):
    """This function takes a point (x and y), an image, and samples the average color in a radius of 'radius' pixels around the point.
    It then decides if it's either yellow, red, blue or green and returns this decision.

    Args:
        point (list): list of x and y coordinates
        image (np.ndarray): Image to sample from
        sample_size (int, optional): Square region of interes with length of radius. Defaults to 10.

    Returns:
        string: Name of the sampled color. Either yellow, red, blue or green.
    """
    
    x = int(point[0])
    y = int(point[1])
    # Calculate the coordinates of the region of interest
    x_min = max(x - sample_size, 0)
    y_min = max(y - sample_size, 0)
    x_max = min(x + sample_size, image.shape[1] - 1)
    y_max = min(y + sample_size, image.shape[0] - 1)
    
    # Extract the region of interest
    roi = image[y_min:y_max+1, x_min:x_max+1]
    
    # Calculate the average color of the region of interest
    avg_color = np.mean(roi, axis=(0, 1)).astype(int)
    
    # Define the color thresholds for each color
    yellow_threshold = np.array([0, 250, 250])
    red_threshold = np.array([0, 0, 255])
    blue_threshold = np.array([255, 0, 0])
    green_threshold = np.array([0, 240, 0])
    
    # Calculate the distance between the average color and each color threshold
    distances = [
        np.linalg.norm(avg_color - yellow_threshold),
        np.linalg.norm(avg_color - red_threshold),
        np.linalg.norm(avg_color - blue_threshold),
        np.linalg.norm(avg_color - green_threshold)
    ]
    
    # Decide which color the average color is closest to
    color_decision = np.argmin(distances)
    
    # Return the color decision
    if color_decision == 0:
        return "yellow"
    elif color_decision == 1:
        return "red"
    elif color_decision == 2:
        return "blue"
    elif color_decision == 3:
        return "green"