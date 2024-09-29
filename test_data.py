import cv2
import numpy as np


def add_gaussian_noise(image, mean=0, std=25):
    """
    Adds Gaussian noise to an image.

    Parameters:
    - image: The original image as a NumPy array.
    - mean: Mean of the Gaussian noise (default 0).
    - std: Standard deviation of the Gaussian noise (default 25).

    Returns:
    - noisy_image: The image with added Gaussian noise.
    """
    # Generate Gaussian noise
    gauss = np.random.normal(mean, std, image.shape).astype(np.float32)

    # Add the noise to the image
    noisy_image = image.astype(np.float32) + gauss

    # Clip the values to stay within valid pixel range [0, 255] and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    Adds salt-and-pepper noise to an image.

    Parameters:
    - image: The original image as a NumPy array.
    - salt_prob: Probability of adding 'salt' noise (white pixels).
    - pepper_prob: Probability of adding 'pepper' noise (black pixels).

    Returns:
    - noisy_image: The image with added salt-and-pepper noise.
    """
    noisy_image = np.copy(image)

    # Add salt (white pixels)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Add pepper (black pixels)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image


def add_speckle_noise(image):
    """
    Adds speckle noise to an image.

    Parameters:
    - image: The original image as a NumPy array.

    Returns:
    - noisy_image: The image with added speckle noise.
    """
    noise = np.random.randn(*image.shape).astype(np.float32)  # Generate speckle noise
    noisy_image = image + image * noise  # Apply speckle noise

    # Clip the values to stay within valid pixel range [0, 255] and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


def test_with_noise(image_path, noise_type='gaussian'):
    # Load the image
    image = cv2.imread(image_path)

    # Apply noise
    if noise_type == 'gaussian':
        noisy_image = add_gaussian_noise(image)
    elif noise_type == 'salt_and_pepper':
        noisy_image = add_salt_and_pepper_noise(image)
    elif noise_type == 'speckle':
        noisy_image = add_speckle_noise(image)
    else:
        print("Unknown noise type")
        return

    # Save and display the noisy image
    cv2.imwrite('noisy_image.png', noisy_image)
    # cv2.imshow('Noisy Image', noisy_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Run the bounding box detection on the noisy image
    # draw_bounding_box_non_white('noisy_image.png')


def convert_to_grayscale(image):
    """
    Convert the input image to grayscale.

    Parameters:
    - image: The original input image.

    Returns:
    - gray_image: The grayscale image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def convert_to_binary(image, threshold=128):
    """
    Convert the input image to binary (black and white).

    Parameters:
    - image: The input image (preferably grayscale).
    - threshold: The threshold value to convert the image to binary (default is 128).

    Returns:
    - binary_image: The binary image.
    """
    # If image is not already in grayscale, convert it
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def apply_canny_edge_detection(image, lower_threshold=100, upper_threshold=200):
    """
    Apply Canny edge detection on the input image.

    Parameters:
    - image: The input image (preferably grayscale).
    - lower_threshold: The lower threshold for edge detection.
    - upper_threshold: The upper threshold for edge detection.

    Returns:
    - edges: The image with detected edges.
    """
    # If image is not already in grayscale, convert it
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(image, lower_threshold, upper_threshold)
    return edges


# Example usage

# file_path = 'test_images/scanned_document.png'
# test_with_noise(file_path, noise_type='gaussian')
#
# image = cv2.imread('noisy_image.png')
#
# grayscaled_img = convert_to_grayscale(image)
# bw_img = convert_to_binary(grayscaled_img)
#
# cv2.imwrite('bw_image.png', bw_img)
# edge_img = apply_canny_edge_detection(bw_img)

# cv2.imshow("Denoised Image", edge_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def generate():
    pass