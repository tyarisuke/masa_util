import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Module for image processing
# It can use in black and white images or color images


def read(filepath, flags=cv2.IMREAD_COLOR):
    """
    Wrap cv2.imread function to simplify its usage and add custom behavior if needed.

    Args:
    - filepath (str): The path to the image file to be read.
    - flags (int): Flag that specifies the way the image should be read. Default is cv2.IMREAD_COLOR.

    Returns:
    - numpy.ndarray: The image read from the file.
    """
    image = cv2.imread(filepath, flags)
    return image


def write(filepath, image, params=None):
    """
    Wrap cv2.imwrite function to simplify its usage and add custom behavior if needed.

    Args:
    - filepath (str): The path where the image will be saved.
    - image (numpy.ndarray): The image to be saved.
    - params (list of int): Optional parameters for the specific format.

    Returns:
    - bool: True if the image is saved successfully, otherwise False.
    """
    result = (
        cv2.imwrite(filepath, image, params)
        if params
        else cv2.imwrite(filepath, image)
    )
    return result


def pil_to_opencv(image):
    """
    Convert a PIL Image to an OpenCV image format in BGR color space.

    Args:
    - image (PIL.Image.Image): The image loaded by PIL.

    Returns:
    - numpy.ndarray: The image in OpenCV format (BGR).
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def pil_to_opencv_gray(image):
    """
    Convert a PIL Image to an OpenCV image format in grayscale.

    Args:
    - image (PIL.Image.Image): The image loaded by PIL.

    Returns:
    - numpy.ndarray: The image in OpenCV format (grayscale).
    """
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)


def opencv_to_pil(opencv_image):
    """
    Convert an OpenCV image format in BGR color space to a PIL Image.

    Args:
    - opencv_image (numpy.ndarray): The image in OpenCV format (BGR).

    Returns:
    - PIL.Image.Image: The converted image in PIL format.
    """
    rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def get_image_size(image):
    """
    Retrieve the dimensions (width, height) of an image.

    Args:
    - image (numpy.ndarray): The image whose size is to be retrieved.

    Returns:
    - tuple: The width and height of the image.
    """
    height, width = image.shape[:2]
    return (width, height)


def convert_to_grayscale(image):
    """
    Convert a color image to grayscale using OpenCV.

    Args:
    - image (numpy.ndarray): The color image to convert.

    Returns:
    - numpy.ndarray: The grayscale image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def show_image(
    image, window_name="Image", wait_time=0, close_window=True, scale=None
):
    """
    Display an image using OpenCV, with options for scaling, display duration, and window closing behavior.

    Args:
    - image (numpy.ndarray): The image to display, can be color or grayscale.
    - window_name (str, optional): The name of the window in which the image will be displayed. Default is "Image".
    - wait_time (int, optional): The amount of time (in milliseconds) to wait for a key press before automatically continuing.
                                 If set to 0, it will wait indefinitely until a key is pressed. Default is 0.
    - close_window (bool, optional): Flag to determine whether to close all windows after displaying the image. Default is True.
    - scale (float, optional): Scale factor for resizing the image. If None, no scaling is applied. Default is None.

    Examples:
    - To display the image at half size:
      show_image(my_image, scale=0.5)
    - To display the image at double size:
      show_image(my_image, scale=2.0)
    """
    if scale is not None:
        # Calculate the new dimensions
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        # Resize the image
        resized_image = cv2.resize(image, (width, height))
        image_to_show = resized_image
    else:
        image_to_show = image

    cv2.imshow(window_name, image_to_show)
    cv2.waitKey(
        wait_time
    )  # Wait for the specified time or until a key is pressed

    if close_window:
        cv2.destroyAllWindows()  # Close all OpenCV windows


def template_matching(main_image, template, region=None):
    """
    Perform template matching to identify the location of the template within a specific region of the main image,
    or in the entire image if no region is specified. Ensures the template is smaller than the region or image.

    Args:
    - main_image (numpy.ndarray): The main image where the search is performed.
    - template (numpy.ndarray): The template image to find within the main image.
    - region (tuple, optional): A tuple (x, y, w, h) specifying the top-left corner and the width and height of the search region.

    Returns:
    - tuple: Coordinates of the top-left and bottom-right corner of the matched region, match value, and the image with highlighted match.
    """
    if len(main_image.shape) > 2:
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    else:
        main_gray = main_image

    if len(template.shape) > 2:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template

    if region is None:
        roi = main_gray
        offset_x, offset_y = 0, 0
    else:
        x, y, w, h = region
        if template_gray.shape[0] > h or template_gray.shape[1] > w:
            raise ValueError(
                "Template must be smaller than the region of interest."
            )
        roi = main_gray[y : y + h, x : x + w]
        offset_x, offset_y = x, y

    template_w, template_h = template_gray.shape[::-1]

    # Perform matching within the region of interest or the whole image
    res = cv2.matchTemplate(roi, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    # Adjust max_loc based on the offset
    top_left = (max_loc[0] + offset_x, max_loc[1] + offset_y)
    bottom_right = (top_left[0] + template_w, top_left[1] + template_h)

    matched_image = cv2.rectangle(
        main_image.copy(), top_left, bottom_right, (0, 255, 0), 2
    )

    return (top_left, bottom_right), max_val, matched_image


def count_template_matches(main_image, template, threshold=0.8):
    """
    Counts the number of times a template image appears in a main image using template matching.
    The function automatically converts color images to grayscale for the matching process.
    A threshold is used to determine matches, where a higher threshold results in stricter matching.

    Args:
    - main_image (np.ndarray): The main image in which to find the template. Can be grayscale or color.
    - template (np.ndarray): The template image to match within the main image. Can be grayscale or color.
    - threshold (float, optional): The threshold for matching, between 0 and 1. Default is 0.8.

    Returns:
    - int: The number of times the template was found in the main image.
    """
    # Ensure images are in grayscale for matching
    if len(main_image.shape) > 2:
        main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) > 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Perform template matching
    res = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)

    # Find locations of matches above the threshold
    loc = np.where(res >= threshold)

    # Count unique matches
    count = len(set(zip(*loc[::-1])))

    return count


def draw_circle(image, center, radius, color=(0, 255, 0), thickness=2):
    """
    Draw a circle on an image using OpenCV. This function supports both grayscale and color images.

    Args:
    - image (numpy.ndarray): The image on which to draw the circle.
    - center (tuple of int): The (x, y) coordinates of the center of the circle.
    - radius (int): The radius of the circle.
    - color (tuple, optional): The color of the circle in BGR format for color images, or grayscale value for grayscale images. Default is green (0, 255, 0) for color images.
    - thickness (int, optional): The thickness of the circle's outline. If set to -1, the circle will be filled. Default is 2.

    Returns:
    - numpy.ndarray: The image with the circle drawn on it.
    """
    # Check if the image is grayscale (2 dimensions) or color (3 dimensions)
    if len(image.shape) == 2:  # Grayscale image
        # Ensure the color is a single integer if the image is grayscale
        grayscale_color = 255 if len(color) > 1 else color[0]
        cv2.circle(image, center, radius, grayscale_color, thickness)
    else:  # Color image
        cv2.circle(image, center, radius, color, thickness)

    return image


def merge_images_max_pixel(image1, image2):
    """
    Merge two images by taking the maximum pixel value at each position.

    Args:
    - image1 (numpy.ndarray): The first image.
    - image2 (numpy.ndarray): The second image.

    Returns:
    - numpy.ndarray: Merged image with the maximum pixel values.

    Raises:
    - ValueError: If the input images do not have the same dimensions.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions to merge.")

    # Calculate the maximum pixel values across the corresponding pixels in both images
    merged_image = np.maximum(image1, image2)

    return merged_image


def pad_image(image, top, bottom, left, right, value=0):
    """
    Pad an image with specified values on all sides.

    Args:
    - image (numpy.ndarray): The image to pad.
    - top (int): The number of pixels to pad on the top.
    - bottom (int): The number of pixels to pad on the bottom.
    - left (int): The number of pixels to pad on the left.
    - right (int): The number of pixels to pad on the right.
    - value (int or tuple, optional): The value used for padding if border_type is cv2.BORDER_CONSTANT. Default is 0.

    Returns:
    - numpy.ndarray: The padded image.
    """
    # Apply padding to the image
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
    )

    return padded_image


def otsu_threshold(image):
    """
    Apply Otsu's thresholding method to an image to separate the foreground from the background.

    Args:
    - image (numpy.ndarray): The image on which to apply the thresholding.

    Returns:
    - numpy.ndarray: The binary image result after applying Otsu's threshold.
    """
    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    _, binary_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary_image


def morphological_dilation(
    image, kernel_size=(5, 5), kernel_shape=cv2.MORPH_RECT
):
    """
    Apply morphological dilation to an image to expand the foreground.

    Args:
    - image (numpy.ndarray): The image on which to perform the operation.
    - kernel_size (tuple of int, optional): The size of the kernel used for the morphological operation. Default is (5, 5).
    - kernel_shape (int, optional): The shape of the structuring element. Options include cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, and cv2.MORPH_CROSS. Default is cv2.MORPH_RECT.

    Returns:
    - numpy.ndarray: The image after applying morphological dilation.
    """
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    dilated_image = cv2.dilate(image, kernel)
    return dilated_image


def morphological_erosion(
    image, kernel_size=(5, 5), kernel_shape=cv2.MORPH_RECT
):
    """
    Apply morphological erosion to an image to shrink the foreground.

    Args:
    - image (numpy.ndarray): The image on which to perform the operation.
    - kernel_size (tuple of int, optional): The size of the kernel used for the morphological operation. Default is (5, 5).
    - kernel_shape (int, optional): The shape of the structuring element. Options include cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, and cv2.MORPH_CROSS. Default is cv2.MORPH_RECT.

    Returns:
    - numpy.ndarray: The image after applying morphological erosion.
    """
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    eroded_image = cv2.erode(image, kernel)
    return eroded_image


def morphological_opening(
    image, kernel_size=(5, 5), kernel_shape=cv2.MORPH_RECT
):
    """
    Apply morphological opening to an image to remove small noise.

    Args:
    - image (numpy.ndarray): The image on which to perform the operation. Can be a color or grayscale image.
    - kernel_size (tuple of int, optional): The size of the kernel used for the morphological operation. Default is (5, 5).
    - kernel_shape (int, optional): The shape of the structuring element. Options include cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS. Default is cv2.MORPH_RECT.

    Returns:
    - numpy.ndarray: The image after applying morphological opening.
    """
    # Create the kernel (structuring element)
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    # Apply the opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    return opened_image


def morphological_closing(
    image, kernel_size=(5, 5), kernel_shape=cv2.MORPH_RECT
):
    """
    Apply morphological closing to an image to fill small holes and gaps.

    Args:
    - image (numpy.ndarray): The image on which to perform the operation.
    - kernel_size (tuple of int, optional): The size of the kernel used for the morphological operation. Default is (5, 5).
    - kernel_shape (int, optional): The shape of the structuring element. Options include cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, and cv2.MORPH_CROSS. Default is cv2.MORPH_RECT.

    Returns:
    - numpy.ndarray: The image after applying morphological closing.
    """
    # Create the kernel (structuring element)
    kernel = cv2.getStructuringElement(kernel_shape, kernel_size)
    # Apply the closing operation
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return closed_image


def find_bounding_boxes_within_range(
    image, min_width, min_height, max_width, max_height
):
    """
    Find bounding boxes of objects within specified width and height ranges in a binary image.

    Args:
    - image (numpy.ndarray): The binary image from which to find the bounding boxes.
    - min_width (int): Minimum width of the bounding boxes.
    - min_height (int): Minimum height of the bounding boxes.
    - max_width (int): Maximum width of the bounding boxes.
    - max_height (int): Maximum height of the bounding boxes.

    Returns:
    - list of tuples: A list containing the coordinates (x, y, width, height) of each bounding box that meets the criteria.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # List to store bounding boxes that meet the criteria
    valid_bounding_boxes = []

    # Loop through the contours
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)

        # Check if the bounding box meets the specified size requirements
        if (
            min_width <= width <= max_width
            and min_height <= height <= max_height
        ):
            valid_bounding_boxes.append((x, y, width, height))

    return valid_bounding_boxes


def draw_bounding_boxes(image, boxes, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on an image. Supports both color and grayscale images.

    Args:
    - image (numpy.ndarray): The image on which to draw the bounding boxes.
    - boxes (list of tuples): A list of tuples, where each tuple contains the coordinates and size of the box (x, y, width, height).
    - color (tuple, optional): The color of the bounding boxes in BGR format for color images, or grayscale value for grayscale images. Default is green (0, 255, 0) for color images.
    - thickness (int, optional): The thickness of the bounding box's outline. If set to -1, the box will be filled. Default is 2.

    Returns:
    - numpy.ndarray: The image with bounding boxes drawn on it.
    """
    # Check if the image is grayscale (2 dimensions) or color (3 dimensions)
    if len(image.shape) == 2:  # Grayscale image
        # Ensure the color is a single integer if the image is grayscale
        grayscale_color = 255 if len(color) > 1 else color[0]
        color = grayscale_color  # Update color to be the grayscale color

    # Make a copy of the image to draw on
    image_with_boxes = image.copy()

    # Draw each bounding box
    for x, y, width, height in boxes:
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        cv2.rectangle(
            image_with_boxes, top_left, bottom_right, color, thickness
        )

    return image_with_boxes


def subtract_images(image1, image2):
    """
    Subtract the pixel values of image2 from image1 and return the result.

    Args:
    - image1 (numpy.ndarray): The first image (minuend).
    - image2 (numpy.ndarray): The second image (subtrahend).

    Returns:
    - numpy.ndarray: The result of the subtraction.

    Raises:
    - ValueError: If the input images do not have the same dimensions.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            "Images must have the same dimensions to perform subtraction."
        )

    # Subtract image2 from image1 using OpenCV, which automatically handles underflow by clipping at 0
    result_image = cv2.subtract(image1, image2)

    return result_image


def find_white_centroid(image):
    """
    Find the centroid of the white regions in an image.

    Args:
    - image (numpy.ndarray): Input image, which should be a binary image where white represents the areas of interest.

    Returns:
    - tuple: The (x, y) coordinates of the centroid of the white areas, or None if no white area is found.
    """
    # Convert the image to grayscale if it is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Assume the image is already thresholded such that white areas are of interest
    # If not, apply thresholding:
    # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find all white pixels
    white_pixels = np.where(image == 255)

    if white_pixels[0].size == 0:
        # No white pixels found
        return None

    # Compute the centroid of white pixels
    y_centroid = np.mean(white_pixels[0])
    x_centroid = np.mean(white_pixels[1])

    return (int(x_centroid), int(y_centroid))


def binarize_image(image, threshold=127, use_otsu=False):
    """
    Binarize an image using a fixed threshold or Otsu's thresholding.

    Args:
    - image (numpy.ndarray): Input image, which can be a grayscale or color image.
    - threshold (int, optional): The threshold value used for binary thresholding. Default is 127.
    - use_otsu (bool, optional): Whether to use Otsu's thresholding method instead of a fixed threshold. Default is False.

    Returns:
    - numpy.ndarray: The binarized image.
    """
    # Convert the image to grayscale if it is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    if use_otsu:
        # Use Otsu's thresholding
        _, binary_image = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        # Use a fixed threshold
        _, binary_image = cv2.threshold(
            image, threshold, 255, cv2.THRESH_BINARY
        )

    return binary_image


def cumulative_horizontal_histogram(image, threshold=128):
    """
    Calculate the cumulative horizontal histogram of bright (white) pixels in an image.

    Args:
    - image (numpy.ndarray): The input image (grayscale or color).
    - threshold (int, optional): The pixel value threshold to consider as 'bright'. Default is 255.

    Returns:
    - numpy.ndarray: The cumulative horizontal histogram of bright pixels.
    """
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Threshold the image to get bright areas
    _, binary_image = cv2.threshold(
        gray_image, threshold, 255, cv2.THRESH_BINARY
    )

    # Sum bright pixels horizontally
    horizontal_sum = np.sum(binary_image == 255, axis=0)

    # Calculate the cumulative sum of these bright pixels horizontally
    cumulative_horizontal_sum = np.cumsum(horizontal_sum)

    return cumulative_horizontal_sum


def compare_images(image1, image2):
    """
    Calculate the similarity between two images using the Structural Similarity Index (SSIM).

    Args:
    - image1 (numpy.ndarray): The first image.
    - image2 (numpy.ndarray): The second image.

    Returns:
    - float: The SSIM index between the two images.
    """
    # Convert images to grayscale
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the smaller of the two images
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    min_height = min(h1, h2)
    min_width = min(w1, w2)

    # Resize both images to the size of the smaller image
    image1_resized = cv2.resize(image1, (min_width, min_height))
    image2_resized = cv2.resize(image2, (min_width, min_height))

    # Compute the SSIM between the two resized images
    score, _ = ssim(image1_resized, image2_resized, full=True)
    return score


def crop_circle(image, center, radius):
    """
    Crop a circular region from an image.

    Args:
    - image (numpy.ndarray): The image from which to crop the circle.
    - center (tuple of int): The (x, y) coordinates of the center of the circle.
    - radius (int): The radius of the circle.

    Returns:
    - numpy.ndarray: The cropped circular image.

    Raises:
    - ValueError: If the radius is larger than the image dimensions allow.
    """
    if radius > min(
        center[0],
        center[1],
        image.shape[1] - center[0],
        image.shape[0] - center[1],
    ):
        raise ValueError(
            "Radius is too large for the given center and image dimensions."
        )

    # Create a mask with the same dimensions as the image, initialized to zero (black)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw a filled white circle on the mask at the specified center and radius
    cv2.circle(mask, center, radius, (255), thickness=-1)

    # Apply the mask to the image using bitwise AND
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Create a bounding box around the circle to crop to the minimal area
    x, y = center[0] - radius, center[1] - radius
    cropped_image = masked_image[y : y + 2 * radius, x : x + 2 * radius]

    return cropped_image


def crop(image, x_or_coords, y=None, width=None, height=None):
    """
    Crop an image. Supports both color and grayscale images.
    This function can accept parameters as separate values or as tuples.

    Args:
    - image (numpy.ndarray): The image to be cropped.
    - x_or_coords (int or tuple): If int, it is the x coordinate of the crop's top-left corner.
                                  If tuple, it should be ((x, y), (width, height)).
    - y (int, optional): The y coordinate of the top-left corner of the cropping area.
    - width (int, optional): The width of the cropping area.
    - height (int, optional): The height of the cropping area.

    Returns:
    - numpy.ndarray: The cropped image.

    Raises:
    - ValueError: If the cropping area is out of bounds of the image dimensions or parameters are invalid.
    """
    # Check if coordinates and dimensions are passed as tuples
    if (
        isinstance(x_or_coords, tuple)
        and y is None
        and width is None
        and height is None
    ):
        ((x, y), (width, height)) = x_or_coords
    elif (
        isinstance(x_or_coords, int)
        and isinstance(y, int)
        and isinstance(width, int)
        and isinstance(height, int)
    ):
        x = x_or_coords
    else:
        raise ValueError("Invalid parameters for cropping.")

    # Check if the crop rectangle is within the image bounds
    if (
        x < 0
        or y < 0
        or x + width > image.shape[1]
        or y + height > image.shape[0]
    ):
        raise ValueError("Cropping area is out of the image bounds.")

    # Perform the cropping
    cropped_image = image[y : y + height, x : x + width]
    return cropped_image


def find_image_differences(image1, image2):
    """
    Compare two images (either color or grayscale), find the difference, and return the difference image and the number of differing pixels.

    Args:
    - image1 (numpy.ndarray): First image for comparison.
    - image2 (numpy.ndarray): Second image for comparison.

    Returns:
    - numpy.ndarray: Image showing the differences.
    - int: Number of pixels that are different.
    """
    # Check if the images are color (3 channels) and convert to grayscale if necessary
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1

    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2

    # Compute the absolute difference between the two images
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the diff image to get the foreground (difference)
    _, thresh = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Count non-zero pixels (differences)
    non_zero_count = np.count_nonzero(thresh)

    return thresh, non_zero_count


def detect_circles(
    image,
    dp=1.5,
    min_dist=50,
    param1=100,
    param2=30,
    min_radius=0,
    max_radius=0,
):
    """
    Detect circles in an image using the Hough Circle Transform.

    Args:
    - image (numpy.ndarray): The input image on which circles are to be detected. It must be a grayscale image.
    - dp (float): Inverse ratio of the accumulator resolution to the image resolution. Default is 1.5.
    - min_dist (int): Minimum distance between the centers of the detected circles. Default is 50.
    - param1 (int): First method-specific parameter. In case of using CV_HOUGH_GRADIENT, it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller). Default is 100.
    - param2 (int): Second method-specific parameter. In case of CV_HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first. Default is 30.
    - min_radius (int): Minimum circle radius. Default is 0.
    - max_radius (int): Maximum circle radius. Default is 0.

    Returns:
    - circles (list of tuples): List of tuples where each tuple contains (x, y, radius) for each detected circle.
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise and improve the detection process
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp,
        min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    # Ensure some circles were found
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        return [(x, y, r) for x, y, r in circles]
    else:
        return []  # Return an empty list if no circles were found


def convert_x_y_w_h(top_left, bottom_right):
    """
    Convert coordinates from top-left and bottom-right points to x, y, width, and height format.

    Args:
    - top_left (tuple): A tuple (t, l) representing the top-left corner coordinates of a rectangle.
    - bottom_right (tuple): A tuple (b, r) representing the bottom-right corner coordinates of a rectangle.

    Returns:
    - tuple: A tuple (x, y, w, h) where x and y represent the coordinates of the top-left corner,
             w is the width of the rectangle, and h is the height. This format is often used for specifying
             bounding boxes in various image processing tasks.

    Example:
    - Convert from two corner coordinates to bounding box format:
      >>> top_left = (10, 20)
      >>> bottom_right = (30, 50)
      >>> convert_x_y_w_h(top_left, bottom_right)
      (10, 20, 20, 30)
    """
    t, l = top_left
    b, r = bottom_right
    x, y, w, h = t, l, b - t, r - l
    return x, y, w, h


def paste(large_image, small_image, x_offset, y_offset):
    """
    Paste a small image onto a specified position of a large image using OpenCV.

    Args:
    - large_image (numpy.ndarray): The larger image on which the small image will be pasted.
    - small_image (numpy.ndarray): The small image to paste.
    - x_offset (int): The x-coordinate of the top-left corner where the small image will be pasted.
    - y_offset (int): The y-coordinate of the top-left corner where the small image will be pasted.

    Returns:
    - numpy.ndarray: The large image with the small image pasted onto it.

    Raises:
    - ValueError: If the small image goes beyond the bounds of the large image.
    """
    # Check if the small image fits within the large image at the given offsets
    if (y_offset + small_image.shape[0] > large_image.shape[0]) or (
        x_offset + small_image.shape[1] > large_image.shape[1]
    ):
        raise ValueError(
            "The small image extends beyond the bounds of the large image."
        )

    # Paste the small image onto the large image
    large_image[
        y_offset : y_offset + small_image.shape[0],
        x_offset : x_offset + small_image.shape[1],
    ] = small_image

    return large_image


def create_black(width, height, color=True):
    """
    Create a black image of specified width and height.

    Args:
    - width (int): The width of the image.
    - height (int): The height of the image.
    - color (bool, optional): If True, creates a color image. If False, creates a grayscale image. Default is True.

    Returns:
    - numpy.ndarray: The created black image.
    """
    if color:
        # Create a black color image (3 channels)
        return np.zeros((height, width, 3), dtype=np.uint8)
    else:
        # Create a black grayscale image (1 channel)
        return np.zeros((height, width), dtype=np.uint8)


def concatenate_horizontally(image1, image2):
    """
    Concatenate two images horizontally.

    Args:
    - image1 (numpy.ndarray): First image to concatenate.
    - image2 (numpy.ndarray): Second image to concatenate.

    Returns:
    - numpy.ndarray: A new image resulting from horizontal concatenation of the two images.

    Raises:
    - ValueError: If the heights of the images do not match.
    """
    if image1.shape[0] != image2.shape[0]:
        raise ValueError("Heights of images do not match.")

    return np.hstack((image1, image2))


def concatenate_vertically(image1, image2):
    """
    Concatenate two images vertically.

    Args:
    - image1 (numpy.ndarray): First image to concatenate.
    - image2 (numpy.ndarray): Second image to concatenate.

    Returns:
    - numpy.ndarray: A new image resulting from vertical concatenation of the two images.

    Raises:
    - ValueError: If the widths of the images do not match.
    """
    if image1.shape[1] != image2.shape[1]:
        raise ValueError("Widths of images do not match.")

    return np.vstack((image1, image2))


def paste_center(large_image, small_image):
    """
    Paste a small image onto the center of a large image using OpenCV, scaling it down if necessary while maintaining aspect ratio.

    Args:
    - large_image (numpy.ndarray): The larger image on which the small image will be pasted.
    - small_image (numpy.ndarray): The small image to paste.

    Returns:
    - numpy.ndarray: The large image with the small image pasted onto its center.

    Raises:
    - ValueError: If the small image is larger than the large image in any dimension.
    """
    # Get dimensions of both images
    large_height, large_width = large_image.shape[:2]
    small_height, small_width = small_image.shape[:2]

    # Calculate scale to fit the small image within the large image
    scale_width = large_width / small_width
    scale_height = large_height / small_height
    scale = min(scale_width, scale_height)

    # If scaling is needed (scale < 1), resize the small image
    if scale < 1:
        new_width = int(small_width * scale)
        new_height = int(small_height * scale)
        small_image = cv2.resize(small_image, (new_width, new_height))
    elif scale > 1:
        # If no scaling is needed, use the small image as it is
        new_width, new_height = small_width, small_height

    # Calculate the center offset
    y_offset = (large_height - new_height) // 2
    x_offset = (large_width - new_width) // 2

    # Paste the small image onto the large image
    large_image[
        y_offset : y_offset + new_height, x_offset : x_offset + new_width
    ] = small_image

    return large_image


if __name__ == "__main__":
    main_image_path = "../image/sample1.jpg"
    template_image_path = "../image/template.jpg"

    main_image = cv2.imread(main_image_path)
    template_image = cv2.imread(template_image_path)
    top_left, score, matched_image = template_matching(
        main_image, template_image
    )
    print(f"Match Score: {score}")
    cv2.imshow("Matched Image", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
