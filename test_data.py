import glob
import random
import cv2
import cairosvg
import numpy as np
import os
from PIL import Image
import io


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

def transparent_to_white(img):
    # Check if the image has an alpha channel
    if img.shape[2] == 4:  # If the image has an alpha channel
        # Create an output image with a white background
        white_background = np.ones_like(img, dtype=np.uint8) * 245  # Create a white image

        # Split the channels
        b, g, r, a = cv2.split(img)  # Separate the channels including alpha

        # Create a mask from the alpha channel
        alpha_mask = a / 255.0  # Normalize alpha values to [0, 1]

        # Prepare an output image with white background
        output_image = np.zeros_like(img)  # Initialize output image

        # Blend the original image with the white background
        for c in range(3):  # For each channel (B, G, R)
            output_image[:, :, c] = (alpha_mask * img[:, :, c] + (1 - alpha_mask) * white_background[:, :, c])

        # Set alpha channel to 255 (opaque) for the output image
        output_image[:, :, 3] = 255

        return output_image  # Return the modified image
    else:
        print("The image does not have an alpha channel.")
        return img  # Return the original image if no alpha channel

def resize_image(image, max_size=200):
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Determine new dimensions while maintaining aspect ratio
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def rgba_to_rgb(img):
    # Check if the image has an alpha channel
    if img.shape[2] == 4:  # If the image has an alpha channel
        # Convert RGBA to RGB by ignoring the alpha channel
        rgb_image = img[:, :, :3]  # Take only the first three channels (R, G, B)
        return rgb_image
    else:
        print("The image does not have an alpha channel.")
        return img  # Return the original image if no alpha channel


def svg_to_png_opencv(svg_file_path, png_width=None, png_height=None):
    """
    Convert an SVG file to a 32-bit PNG image using OpenCV.

    Parameters:
        svg_file_path (str): The path to the SVG file.
        png_width (int): Optional width of the output PNG image.
        png_height (int): Optional height of the output PNG image.

    Returns:
        numpy.ndarray: The 32-bit PNG image with an alpha channel.
    """
    # Read the SVG file
    with open(svg_file_path, 'rb') as svg_file:
        svg_data = svg_file.read()

    # If dimensions are specified, apply the width and height to cairosvg
    if png_width is not None and png_height is not None:
        png_data = cairosvg.svg2png(bytestring=svg_data, output_width=png_width, output_height=png_height)
    else:
        png_data = cairosvg.svg2png(bytestring=svg_data)

    # Convert the PNG data to an OpenCV image
    png_image = np.frombuffer(png_data, dtype=np.uint8)
    png_image = cv2.imdecode(png_image, cv2.IMREAD_UNCHANGED)  # Load with alpha channel (32-bit)

    # Ensure the output image is in 32-bit (RGBA format)
    if png_image.shape[2] == 3:  # If it's RGB, convert to RGBA
        png_image = cv2.cvtColor(png_image, cv2.COLOR_RGB2RGBA)

    return png_image

# Function to create a grid of an image with padding using OpenCV
def create_image_grid(input_image_path, output_image_path, grid_size=(3, 3), padding=10, input_image=None):
    # Read the image using OpenCV
    if input_image is not None:
        img = input_image
    else:
        img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

    img = transparent_to_white(img)
    img = resize_image(img)
    # Add padding around the image (optional)
    if padding > 0:
        img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[245, 245, 245])

    # Get the size of the padded image
    img_height, img_width, _ = img.shape

    # Create a blank canvas for the grid
    grid_width = img_width * grid_size[0]  # Total width (number of columns)
    grid_height = img_height * grid_size[1]  # Total height (number of rows)
    grid_img = np.ones((grid_height, grid_width, 4), dtype=np.uint8) * 255  # White background

    # Loop through the grid and place the image in each cell
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            # Calculate the position for each image in the grid
            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width

            # Paste the image into the grid
            grid_img[y_start:y_end, x_start:x_end] = img

    # Save the final grid image
    grid_img = rgba_to_rgb(grid_img)
    cv2.imwrite(output_image_path, grid_img)
    print(f"Saved grid image: {output_image_path}")


# Function to process all images in a folder
def process_images_in_folder(input_folder, output_folder, grid_size=(3, 3), padding=10):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, f'grid_{filename}')

            # Create a grid for the current image
            create_image_grid(input_image_path, output_image_path, grid_size, padding)


def list_files_w_ext(directory, ext):
    """
    Lists all files ending in .(ext) in the specified directory using glob.

    Parameters:
        directory (str): The path to the directory.
        ext (str): Search for files with this extension.

    Returns:
        list: A list of .(ext) file paths.
    """
    return glob.glob(os.path.join(directory, '**', f'*.{ext}'), recursive=True)

def process_svgs_in_folder(input_folder, output_folder, data_set_size=None):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all svg logos
    svg_logos = list_files_w_ext(input_folder, "svg")

    # Randomly select data_set_size from logos
    if data_set_size is not None and data_set_size < len(svg_logos):
        selected_logos = random.sample(svg_logos, data_set_size)
    else:
        selected_logos = svg_logos

    # Loop over svg images, convert to png, and process into grid
    for logo_path in selected_logos:
        img = svg_to_png_opencv(logo_path, png_width=None, png_height=None)
        img = resize_image(img)
        row = random.choice(range(1, 7))
        col = random.choice(range(1, 4))
        # prevent 1x1
        while row * col == 1:
            row = random.choice(range(1, 7))
            col = random.choice(range(1, 4))
        padding = random.choice(range(20, 51, 5))
        logo_name = os.path.basename(logo_path).split(".")[0]

        output_image_path = os.path.join(output_folder, f"{logo_name}_g{row}x{col}_p{padding}.png")
        create_image_grid("", output_image_path, (col, row), padding, input_image=img)



# Set the grid size (e.g., 3x3, 4x4, etc.) and padding in pixels
# grid_size = (4, 4)  # 4 columns by 4 rows
# padding = 20  # Padding in pixels around each image
#
# # Process all images in the folder
# process_images_in_folder(input_folder, output_folder, grid_size, padding)


# Define input and output folders
input_folder = './test_data/logos'  # Folder containing the original images
output_folder = './test_data/output_images'  # Folder where grid images will be saved

def generate_test_data(n):
    process_svgs_in_folder(input_folder, output_folder, data_set_size=n)