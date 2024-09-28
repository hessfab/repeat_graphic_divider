import cv2
import numpy as np


def draw_dashed_line(img, start, end, dash_length=10, gap_length=5, color=(0, 0, 0), thickness=2):
    """
    Helper function to draw dashed lines using OpenCV.
    """
    x1, y1 = start
    x2, y2 = end
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    num_dashes = int(length // (dash_length + gap_length))

    for i in range(num_dashes + 1):
        # Calculate the start and end of each dash
        start_x = int(x1 + i * (dash_length + gap_length) * (x2 - x1) / length)
        start_y = int(y1 + i * (dash_length + gap_length) * (y2 - y1) / length)
        end_x = int(start_x + dash_length * (x2 - x1) / length)
        end_y = int(start_y + dash_length * (y2 - y1) / length)

        # Draw the dash
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)


def check_line_intersects_boxes(start, end, bounding_boxes):
    """
    Check if the line intersects with any of the bounding boxes.

    Args:
    - start: Tuple (x, y) for the start point of the line.
    - end: Tuple (x, y) for the end point of the line.
    - bounding_boxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...].

    Returns:
    - True if the line intersects any bounding box, False otherwise.
    """
    for box in bounding_boxes:
        box_start = (box[0], box[1])
        box_end = (box[2], box[3])

        # Check if the line intersects the rectangle
        if (start[0] < box[2] and end[0] > box[0] and
                start[1] < box[3] and end[1] > box[1]):
            return True
    return False

def draw_dashed_lines_between_boxes(img, bounding_boxes, dash_length=10, gap_length=5, color=(0, 0, 0),
                                    show_boxes=True, box_color=(0, 0, 255), thickness=2, alignment_tolerance=10):
    """
    Draw both horizontal and vertical dashed lines between bounding boxes using OpenCV, and extend the lines to image edges.
    Test alignment of every combination of bounding boxes with a customizable alignment tolerance.

    Args:
    - image_size: Tuple (width, height) representing the size of the image.
    - bounding_boxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...].
    - dash_length: Length of dashes in the lines.
    - gap_length: Gap between dashes.
    - color: Color of the dashed lines (default is black).
    - show_boxes: If True, draw the bounding boxes (default is True).
    - box_color: Color of the bounding boxes (default is blue in BGR format).
    - thickness: Thickness of the dashed lines and bounding boxes (default is 2).
    - alignment_tolerance: Tolerance for considering two boxes horizontally or vertically aligned (default is 10 pixels).

    Returns:
    - Image with dashed lines and optionally bounding boxes.
    """
    # Create a blank white image
    # img = np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 255

    image_size = (img.shape[1], img.shape[0])

    # Optionally draw bounding boxes
    if show_boxes:
        for box in bounding_boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), box_color, thickness)

    # Check every combination of bounding boxes
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            box1 = bounding_boxes[i]
            box2 = bounding_boxes[j]

            # Check if the boxes are horizontally aligned (same row)
            if abs(box1[1] - box2[1]) < alignment_tolerance:  # Tolerance for horizontal alignment
                # Draw vertical dashed line halfway between the two boxes and extend to image edges
                split_x = (box1[2] + box2[0]) // 2

                # # Extend from top to bottom
                # draw_dashed_line(img, (split_x, 0), (split_x, image_size[1]), dash_length, gap_length, color, thickness)

                # Check if the line intersects with any bounding box
                if not check_line_intersects_boxes((split_x, 0), (split_x, image_size[1]), bounding_boxes):
                    # Extend from top to bottom
                    draw_dashed_line(img, (split_x, 0), (split_x, image_size[1]), dash_length, gap_length, color,
                                     thickness)

            # Check if the boxes are vertically aligned (stacked)
            if abs(box1[0] - box2[0]) < alignment_tolerance:  # Tolerance for vertical alignment
                # Draw horizontal dashed line halfway between the two boxes and extend to image edges
                split_y = (box1[3] + box2[1]) // 2

                # # Extend from left to right
                # draw_dashed_line(img, (0, split_y), (image_size[0], split_y), dash_length, gap_length, color, thickness)

                # Check if the line intersects with any bounding box
                if not check_line_intersects_boxes((0, split_y), (image_size[0], split_y), bounding_boxes):
                    # Extend from left to right
                    draw_dashed_line(img, (0, split_y), (image_size[0], split_y), dash_length, gap_length, color,
                                     thickness)

    return img


# # Example usage
# bounding_boxes = [
#     (50, 50, 150, 150),  # Box 1
#     (200, 50, 300, 150),  # Box 2 (same row as Box 1)
#     (50, 200, 150, 300),  # Box 3 (below Box 1)
#     (200, 200, 300, 300)  # Box 4 (below Box 2)
# ]
#
# # Generate the image with dashed lines and bounding boxes
# image_size = (400, 400)
# img_with_lines_and_boxes = draw_dashed_lines_between_boxes(image_size, bounding_boxes, show_boxes=True,
#                                                            alignment_tolerance=15)
#
# # Save the image
# cv2.imwrite("opencv_dashed_lines_with_tolerance.png", img_with_lines_and_boxes)
#
# # Display the image
# cv2.imshow("Dashed Lines and Boxes", img_with_lines_and_boxes)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
