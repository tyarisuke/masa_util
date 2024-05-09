import math


def angle_between_vectors(o, a, b, direction="clockwise"):
    """
    Calculate the angle in degrees between two vectors, from point O to point A and O to point B,
    in either clockwise or counterclockwise direction.

    Parameters:
    - o (tuple): Coordinates of point O (x, y)
    - a (tuple): Coordinates of point A (x, y)
    - b (tuple): Coordinates of point B (x, y)
    - direction (str): 'clockwise' or 'counterclockwise', direction to calculate the angle

    Returns:
    - float: The angle in degrees between the vectors from O to A and O to B in the specified direction

    Raises:
    - ValueError: If any of the vectors are zero vectors, or calculations are otherwise not possible

    Example:
    - angle_between_vectors((0,0), (1,0), (0,1), 'counterclockwise')  # Result will be 90 degrees
    - angle_between_vectors((0,0), (1,0), (0,1), 'clockwise')        # Result will be 270 degrees
    """
    # Calculate vectors OA and OB
    oa = (a[0] - o[0], a[1] - o[1])
    ob = (b[0] - o[0], b[1] - o[1])

    # Calculate the dot product and norms of the vectors
    dot_product = oa[0] * ob[0] + oa[1] * ob[1]
    norm_oa = math.sqrt(oa[0] ** 2 + oa[1] ** 2)
    norm_ob = math.sqrt(ob[0] ** 2 + ob[1] ** 2)

    # Ensure there is no division by zero
    if norm_oa == 0 or norm_ob == 0:
        raise ValueError(
            "One of the vectors has zero length. Please specify valid points."
        )

    # Calculate the angle using the inverse cosine of the cosine theta
    cos_theta = dot_product / (norm_oa * norm_ob)
    # Correct for numerical errors that might result in cos_theta being outside [-1, 1]
    cos_theta = max(-1, min(1, cos_theta))
    angle = math.acos(cos_theta)

    # Convert angle to degrees
    angle_degrees = math.degrees(angle)

    # Determine angle direction
    if direction == "clockwise":
        # Calculate the cross product z-component to determine the relative orientation
        cross_product_z = oa[0] * ob[1] - oa[1] * ob[0]
        if cross_product_z > 0:
            angle_degrees = 360 - angle_degrees
    elif direction == "counterclockwise":
        # Calculate the cross product z-component to determine the relative orientation
        cross_product_z = oa[0] * ob[1] - oa[1] * ob[0]
        if cross_product_z < 0:
            angle_degrees = 360 - angle_degrees
    else:
        raise ValueError(
            "Invalid direction. Use 'clockwise' or 'counterclockwise'."
        )

    return angle_degrees
