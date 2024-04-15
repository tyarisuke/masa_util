import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


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


def template_matching(main_image, template):
    """
    Perform template matching to identify the location of the template in the main image.

    Args:
    - main_image (numpy.ndarray): The main image where the search is performed.
    - template (numpy.ndarray): The template image to find within the main image.

    Returns:
    - tuple: Coordinates of the top-left corner of the matched region, match value, and the image with highlighted match.
    """
    if len(main_image.shape) > 2:
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    else:
        main_gray = main_image

    if len(template.shape) > 2:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    w, h = template_gray.shape[::-1]

    res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    matched_image = cv2.rectangle(
        main_image.copy(), top_left, bottom_right, (0, 255, 0), 2
    )

    return top_left, max_val, matched_image


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
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary_image


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


def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.

    Args:
    - image (numpy.ndarray): The image on which to draw the bounding boxes.
    - boxes (list of tuples): A list of tuples, where each tuple contains the coordinates and size of the box (x, y, width, height).
    - color (tuple, optional): The color of the bounding boxes in BGR format. Default is green (0, 255, 0).
    - thickness (int, optional): The thickness of the bounding box lines. Default is 2.

    Returns:
    - numpy.ndarray: The image with bounding boxes drawn on it.
    """
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


def crop(image, x, y, width, height):
    """
    Crop an image. This function supports both color and grayscale images.

    Args:
    - image (numpy.ndarray): The image to be cropped. Can be a color (height, width, channels) or grayscale (height, width) image.
    - x, y (int): The coordinates of the top-left corner of the cropping area.
    - width, height (int): The width and height of the cropping area.

    Returns:
    - numpy.ndarray: The cropped image.

    Raises:
    - ValueError: If the cropping area is out of bounds of the image dimensions.
    """
    # Check if the crop rectangle is within the image bounds
    if (
        x < 0
        or y < 0
        or x + width > image.shape[1]
        or y + height > image.shape[0]
    ):
        raise ValueError("Cropping area is out of the image bounds.")

    cropped_image = image[y : y + height, x : x + width]
    return cropped_image


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
