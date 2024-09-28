import cv2
import numpy as np


# Step 1: Load the main image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image


# Step 2: Template matching to find repeating images
def find_repeating_images(main_image, template_image):
    result = cv2.matchTemplate(main_image, template_image, cv2.TM_CCOEFF_NORMED)
    threshold = 0.95  # Adjust threshold as needed to detect identical matches
    locations = np.where(result >= threshold)

    bounding_boxes = []
    w, h = template_image.shape[1], template_image.shape[0]

    for pt in zip(*locations[::-1]):  # Switch x and y coordinates
        bounding_boxes.append((pt, (pt[0] + w, pt[1] + h)))

    return bounding_boxes


# Step 3: Draw bounding boxes on the original image
def draw_bounding_boxes(image, bounding_boxes):
    for top_left, bottom_right in bounding_boxes:
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)  # Red bounding box

    # Save the result
    output_image_path = "output_images/output_with_bboxes.png" # TODO template the file name
    cv2.imwrite(output_image_path, image)
    return output_image_path


# Step 4: Main function to process the image
def process_image(main_image_path, template_image_path):
    main_image = load_image(main_image_path)
    template_image = load_image(template_image_path)

    bounding_boxes = find_repeating_images(main_image, template_image)
    output_image_path = draw_bounding_boxes(main_image, bounding_boxes)

    print(f"Output image saved to: {output_image_path}")
    return bounding_boxes


# # Usage
# main_image_path = "adi_x6.png"  # The image with multiple repeated instances
# template_image_path = "roi.png"  # One of the repeated images
#
# process_image(main_image_path, template_image_path)