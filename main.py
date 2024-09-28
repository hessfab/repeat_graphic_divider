import cv2
import numpy as np

from cutter import draw_dashed_lines_between_boxes
from pattern_detect import process_image


def draw_bounding_box_non_white(image_path, shrink_to, output_path=None,
                                 x_min=0, y_min=0, x_max=None, y_max=None, shrink_by=10):
    """
    Draws a bounding box around non-white pixels in a specified region of the given image.

    Parameters:
    - image_path: str, path to the input image.
    - output_path: str or None, path to save the output image with bounding box.
                    If None, defaults to 'image_with_bounding_box.png'.
    - x_min: int, minimum x coordinate for the region of interest (default is 0).
    - y_min: int, minimum y coordinate for the region of interest (default is 0).
    - x_max: int or None, maximum x coordinate for the region of interest.
              If None, it will be set to the width of the image.
    - y_max: int or None, maximum y coordinate for the region of interest.
              If None, it will be set to the height of the image.
    """
    print("X_min: ", x_min, "Y_min: ", y_min, "X_max: ", x_max, "Y_max: ", y_max)
    if y_max and shrink_to == 'row':
        if y_max <= y_min:
            return None
    if x_max and shrink_to == 'col':
        if x_max <= x_min:
            return None
    # Load the image
    image = cv2.imread(image_path)

    # Set x_max and y_max if not provided
    if x_max is None:
        x_max = image.shape[1]  # Width of the image
    if y_max is None:
        y_max = image.shape[0]  # Height of the image

    # Ensure the coordinates are within the image dimensions
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])

    # Crop the region of interest
    roi = image[y_min:y_max, x_min:x_max]
    # print(roi.shape)

    # Convert the region to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where white pixels are isolated (assuming white is close to 255)
    _, binary_mask = cv2.threshold(gray_roi, 240, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('Binary Mask', binary_mask)

    # Find the coordinates of non-white pixels in the ROI
    y_indices, x_indices = np.where(binary_mask == 255)

    # print("X: ", x_max, x_indices)
    # print("Y: ", y_max, y_indices)
    # if x_max in x_indices:
    #     print("x_max in x_indices")
    # if y_max in y_indices:
    #     print("y_max in y_indices")

    # If there are non-white pixels in the ROI
    if len(x_indices) > 0 and len(y_indices) > 0:
        # Get the bounding box coordinates
        x_min_roi, x_max_roi = np.min(x_indices), np.max(x_indices)
        y_min_roi, y_max_roi = np.min(y_indices), np.max(y_indices)

        # Adjust the coordinates to the original image
        x_min_final = x_min + x_min_roi
        y_min_final = y_min + y_min_roi
        x_max_final = x_min + x_max_roi + 1
        y_max_final = y_min + y_max_roi + 1

        if y_max_final == y_max and shrink_to == 'row':
            print("Y has image in boundary box, shrinking bounding box till white space found")
            return draw_bounding_box_non_white(image_path, shrink_to='row', x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max - shrink_by)
        elif x_max_final == x_max and shrink_to == 'col':
            print("X has image in boundary box, shrinking bounding box till white space found")
            return draw_bounding_box_non_white(image_path, shrink_to='col', x_min=x_min, y_min=y_min, x_max=x_max - shrink_by,
                                               y_max=y_max)

        # if x_max_final == x_max - 1:
        #     print("X has image in boundary box")

        # Draw the bounding box on the original image
        cv2.rectangle(image, (x_min_final-1, y_min_final-1), (x_max_final, y_max_final), (0, 255, 0), 1)

        # Set default output path if not provided
        if output_path is None:
            output_path = 'output_images/image_with_bounding_box.png'

        # Save the image with the bounding box
        cv2.imwrite(output_path, image)
        print(f"Bounding box drawn and saved as '{output_path}'")

        return image, x_min_final, y_min_final, x_max_final, y_max_final

    else:
        print("No non-white pixels found in the specified region of the image.")
        return None





file_path = 'test_images/adi_x12.png'
# file_path = 'noisy_image.png'

# Example usage
image, x_min, y_min, x_max, y_max = draw_bounding_box_non_white(file_path, shrink_to='row') # initially shrink by row

# shrink bounding box to single row
while True:
    data = draw_bounding_box_non_white(
        file_path,
        shrink_to='row',
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=(y_max-y_min)//2 + y_min
        )
    print(data)
    if data is not None:
        image, x_min, y_min, x_max, y_max = data
    else:
        break

# shrink bounding box to single column
while True:
    data = draw_bounding_box_non_white(
        file_path,
        shrink_to='col',
        x_min=x_min,
        y_min=y_min,
        x_max=(x_max - x_min) // 2 + x_min,
        y_max=y_max
    )
    print(data)
    if data is not None:
        image, x_min, y_min, x_max, y_max = data
    else:
        break


print(x_min, y_min, x_max, y_max)

img = cv2.imread(file_path)
roi = img[y_min:y_max, x_min:x_max]
cv2.imwrite("rois/roi.png", roi)

template_image_path = "rois/roi.png"  # One of the repeated images
bounding_boxes = process_image(file_path, template_image_path)
# print(bounding_boxes.shape)
bounding_boxes = np.array(bounding_boxes)
bounding_boxes = bounding_boxes.reshape(bounding_boxes.shape[0], -1)
# print(bounding_boxes.shape)

img_with_lines_and_boxes = draw_dashed_lines_between_boxes(img, bounding_boxes, show_boxes=True,
                                                           alignment_tolerance=15)

# Save the image
cv2.imwrite("opencv_dashed_lines_with_tolerance.png", img_with_lines_and_boxes)

# Display the image
cv2.imshow("Dashed Lines and Boxes", img_with_lines_and_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Optional: Display the image with the bounding box
# cv2.imshow('Image with Bounding Box', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()