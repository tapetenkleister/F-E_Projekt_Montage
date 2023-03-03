import math

def find_closest_corner(center:tuple, corners:list) -> tuple:
    """Checks cornerpoints distance to a given point and the one with the lowest distance is returned.

    Args:
        center (tuple): Centerpoint with x and y coordinate.
        corners (list): List of corners with x and y coordinates.

    Raises:
        ValueError: Multiple closest corners exist.
        ValueError: No corners provided.

    Returns:
        closest_corner: X and Y coordinates of the corner with lowest distance.
    """    
    # Initialize the minimum distance and closest corner
    min_dist = float('inf')
    closest_corner = None
    
    # Iterate over each corner
    for corner in corners:
        # Compute the Euclidean distance to the center
        dist = math.sqrt((corner[0] - center[0])**2 + (corner[1] - center[1])**2)
        
        # Update the closest corner if this distance is smaller than the current minimum
        if dist < min_dist:
            min_dist = dist
            closest_corner = corner
        elif dist == min_dist:
            # There is a tie between two corners
            raise ValueError("Multiple closest corners exist.")
    
    # Check if a closest corner was found
    if closest_corner is None:
        raise ValueError("No corners provided.")
    
    return closest_corner