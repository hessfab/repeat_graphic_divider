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


def consolidate_dashed_lines(hori_lines, vert_lines, threshold):
    if len(hori_lines) > 0:
        hori_lines = consolidate_lines(hori_lines, threshold, is_horizontal=True)
    if len(vert_lines) > 0:
        vert_lines = consolidate_lines(vert_lines, threshold, is_horizontal=False)
    return hori_lines, vert_lines


def consolidate_lines(lines, threshold, is_horizontal=True):
    """
    Consolidates lines (horizontal or vertical) based on proximity.

    :param lines: List of line coordinates in the form (x_min, y_min, x_max, y_max).
    :param threshold: The distance threshold to consolidate lines.
    :param is_horizontal: Whether the lines are horizontal (True) or vertical (False).
    :return: List of consolidated line coordinates.
    """
    # Sort lines based on the relevant axis (y for horizontal, x for vertical)
    if is_horizontal:
        lines.sort(key=lambda line: line[1])  # Sort by y_min (or y_max)
    else:
        lines.sort(key=lambda line: line[0])  # Sort by x_min (or x_max)

    consolidated = []
    group = [lines[0]]  # Start with the first line

    for i in range(1, len(lines)):
        # For horizontal lines, compare y_min (or y_max)
        # For vertical lines, compare x_min (or x_max)
        if is_horizontal:
            if abs(lines[i][1] - group[-1][1]) <= threshold:  # Compare y_min
                group.append(lines[i])
            else:
                # Consolidate the group of horizontal lines
                avg_y = np.mean([line[1] for line in group])  # Average y_min (and y_max)
                x_min = min(line[0] for line in group)
                x_max = max(line[2] for line in group)
                consolidated.append((x_min, avg_y, x_max, avg_y))
                group = [lines[i]]  # Start a new group
        else:
            if abs(lines[i][0] - group[-1][0]) <= threshold:  # Compare x_min
                group.append(lines[i])
            else:
                # Consolidate the group of vertical lines
                avg_x = np.mean([line[0] for line in group])  # Average x_min (and x_max)
                y_min = min(line[1] for line in group)
                y_max = max(line[3] for line in group)
                consolidated.append((avg_x, y_min, avg_x, y_max))
                group = [lines[i]]  # Start a new group

    # Handle the last group
    if is_horizontal:
        avg_y = np.mean([line[1] for line in group])
        x_min = min(line[0] for line in group)
        x_max = max(line[2] for line in group)
        consolidated.append((x_min, avg_y, x_max, avg_y))
    else:
        avg_x = np.mean([line[0] for line in group])
        y_min = min(line[1] for line in group)
        y_max = max(line[3] for line in group)
        consolidated.append((avg_x, y_min, avg_x, y_max))

    return consolidated

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
                                    show_boxes=True, box_color=(0, 0, 255), thickness=2, alignment_tolerance=10, consolidation_threshold=5):
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
    - consolidation_threshold: Dashed lines will be consolidated based on this threshold (default is 5)

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

    # collect coordinates of dashed lines
    vert_dl_coordinates = []
    hori_dl_coordinates = []

    # Check every combination of bounding boxes
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            box1 = bounding_boxes[i]
            box2 = bounding_boxes[j]

            # Check if the boxes are horizontally aligned (same row)
            if abs(box1[1] - box2[1]) < alignment_tolerance:  # Tolerance for horizontal alignment
                # Draw vertical dashed line halfway between the two boxes and extend to image edges
                split_x = (box1[2] + box2[0]) // 2

                # Check if the line intersects with any bounding box
                if not check_line_intersects_boxes((split_x, 0), (split_x, image_size[1]), bounding_boxes):
                    vert_dl_coordinates.append((split_x, 0, split_x, image_size[1]))

            # Check if the boxes are vertically aligned (stacked)
            if abs(box1[0] - box2[0]) < alignment_tolerance:  # Tolerance for vertical alignment
                # Draw horizontal dashed line halfway between the two boxes and extend to image edges
                split_y = (box1[3] + box2[1]) // 2

                # Check if the line intersects with any bounding box
                if not check_line_intersects_boxes((0, split_y), (image_size[0], split_y), bounding_boxes):
                    hori_dl_coordinates.append((0, split_y, image_size[0], split_y))

    # consolidate dashed lines
    hori_dl_coordinates, vert_dl_coordinates = consolidate_dashed_lines(hori_dl_coordinates, vert_dl_coordinates, consolidation_threshold)

    # draw lines
    if len(hori_dl_coordinates) > 0:
        for line in hori_dl_coordinates:
            x0, y0, x1, y1 = line
            # Extend from left to right
            draw_dashed_line(img, (0, y0), (x1, y1), dash_length, gap_length, color, thickness)
    if len(vert_dl_coordinates) > 0:
        for line in vert_dl_coordinates:
            x0, y0, x1, y1 = line
            # Extend from top to bottom
            draw_dashed_line(img, (x0, 0), (x1, y1), dash_length, gap_length, color, thickness)

    return img, hori_dl_coordinates, vert_dl_coordinates
