import cv2
import numpy as np
from PySide2.QtWidgets import QFileDialog
from sklearn.cluster import DBSCAN
from PySide2 import QtWidgets, QtGui, QtCore
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtCore import Qt
import sys

from cutter import draw_dashed_lines_between_boxes
from test_data import convert_to_grayscale, convert_to_binary


class ImageClusterApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Initial display
        self.eps_value = 50

        # Load the image
        self.image_path = 'test_images/adi_x8.png'  # Replace with your image path
        self.image = cv2.imread(self.image_path)

        # Resize the image
        self.resize_image()

        # Calculate total and white pixels
        self.total_pixel_count = self.image.shape[0] * self.image.shape[1]
        grayscaled_img = convert_to_grayscale(self.image)
        self.bw_img = convert_to_binary(grayscaled_img)
        # print(self.image.shape)
        self.white_pixel_count = self.calculate_white_pixels(self.bw_img)

        # Initialize UI
        self.init_ui()

        # Initialize ORB detector
        self.orb = cv2.ORB_create()


        self.update_image()

    def init_ui(self):
        # Create a horizontal layout for the image and controls
        main_layout = QtWidgets.QHBoxLayout()

        # Create a label to display the image
        self.image_label = QtWidgets.QLabel(self)
        main_layout.addWidget(self.image_label)

        # Controls layout
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignTop)
        main_layout.addLayout(controls_layout)

        # Button to open the file explorer and load a new image
        self.open_button = QtWidgets.QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image)
        controls_layout.addWidget(self.open_button)

        # Create a label to display the total number of pixels
        self.total_pixel_label = QtWidgets.QLabel(self)
        self.total_pixel_label.setText(f"Total Pixels: {self.total_pixel_count}")
        controls_layout.addWidget(self.total_pixel_label)

        # Create a label to display the number of white pixels
        self.white_pixel_label = QtWidgets.QLabel(self)
        self.white_pixel_label.setText(f"White Pixels: {self.white_pixel_count}")
        controls_layout.addWidget(self.white_pixel_label)

        # Create a label to display the total area of bounding boxes
        self.total_area_label = QtWidgets.QLabel(self)
        self.total_area_label.setText("Total Area of bboxes: N/A")
        controls_layout.addWidget(self.total_area_label)

        # Create a label to display the calculated value
        self.calculated_value_label = QtWidgets.QLabel(self)
        self.calculated_value_label.setText("Calculated Value: N/A")
        controls_layout.addWidget(self.calculated_value_label)

        # Create a label to display the number of bounding boxes
        self.bbox_count_label = QtWidgets.QLabel(self)
        self.bbox_count_label.setText("# bboxes: N/A")
        controls_layout.addWidget(self.bbox_count_label)

        # Create a label to display the standard deviation
        self.std_label = QtWidgets.QLabel(self)
        self.std_label.setText("ratio(NWpx/Wpx)_std : N/A")
        controls_layout.addWidget(self.std_label)

        # Create a label to display bounding boxes / standard deviation
        self.bbox_std_ratio_label = QtWidgets.QLabel(self)
        self.bbox_std_ratio_label.setText("# bboxes / ratio_std: N/A")
        controls_layout.addWidget(self.bbox_std_ratio_label)

        # Create a label to display the calculated value
        self.eq_maximizer = QtWidgets.QLabel(self)
        self.eq_maximizer.setText("EQ Maximizer: N/A")
        controls_layout.addWidget(self.eq_maximizer)

        # Create a label to display the calculated value
        self.num_divisions = QtWidgets.QLabel(self)
        self.num_divisions.setText("# Divisions: N/A")
        controls_layout.addWidget(self.num_divisions)

        # Create checkboxes for showing/hiding features
        self.show_keypoints_checkbox = QtWidgets.QCheckBox("Show Keypoints", self)
        self.show_keypoints_checkbox.setChecked(False)  # Default not checked
        controls_layout.addWidget(self.show_keypoints_checkbox)
        self.show_keypoints_checkbox.stateChanged.connect(self.update_image)

        self.show_bounding_boxes_checkbox = QtWidgets.QCheckBox("Show Bounding Boxes", self)
        self.show_bounding_boxes_checkbox.setChecked(True)  # Default checked
        controls_layout.addWidget(self.show_bounding_boxes_checkbox)
        self.show_bounding_boxes_checkbox.stateChanged.connect(self.update_image)

        self.show_ratio_area_labels_checkbox = QtWidgets.QCheckBox("Show Ratio/Area Labels", self)
        self.show_ratio_area_labels_checkbox.setChecked(True)  # Default checked
        controls_layout.addWidget(self.show_ratio_area_labels_checkbox)
        self.show_ratio_area_labels_checkbox.stateChanged.connect(self.update_image)

        self.show_divisions_checkbox = QtWidgets.QCheckBox("Show Divisions", self)
        self.show_divisions_checkbox.setChecked(True)  # Default checked
        controls_layout.addWidget(self.show_divisions_checkbox)
        self.show_divisions_checkbox.stateChanged.connect(self.update_image)

        # Create a horizontal layout for the slider and its label
        slider_layout = QtWidgets.QHBoxLayout()

        # Create a slider for adjusting the eps value
        self.eps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.eps_slider.setRange(1, 200)
        self.eps_slider.setValue(self.eps_value)
        self.eps_slider.valueChanged.connect(self.update_image)
        slider_layout.addWidget(self.eps_slider)

        # Create a label to display the current eps value next to the slider
        self.eps_value_label = QtWidgets.QLabel(self)
        self.eps_value_label.setText(f"eps: {self.eps_slider.value()}")
        slider_layout.addWidget(self.eps_value_label)

        controls_layout.addLayout(slider_layout)



        self.setLayout(main_layout)
        self.setWindowTitle("Repeat Graphic Divider - Demo")

    def calculate_white_pixels(self, image):
        # Count the number of white pixels (255, 255, 255)

        # for rgb image
        # white_pixels = np.sum(np.all(image == [255, 255, 255], axis=-1))

        # for bw image
        white_pixels = np.sum(image == 255)
        return white_pixels

    def resize_image(self):
        # TODO resizing may cause discrepancies with coordinates of bboxes and division lines with respect to original image
        # Get the dimensions of the original image
        original_height, original_width = self.image.shape[:2]

        # Define the new width or height (one of them)
        new_height = 800

        # Calculate the ratio of the new width to the original width
        aspect_ratio = new_height / original_height

        # Calculate the new height based on the aspect ratio
        new_width = int(original_width * aspect_ratio)

        # Resize the image while maintaining the aspect ratio
        self.image = cv2.resize(self.image, (new_width, new_height))

    def open_image(self):
        # Open a file dialog to select the image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)")

        if file_path:
            # Load the new image
            self.image_path = file_path
            self.image = cv2.imread(self.image_path)
            self.resize_image()

            if self.image is None:
                QtWidgets.QMessageBox.critical(self, "Error", "Failed to load the image.")
                return

            # Recalculate total and white pixels
            self.total_pixel_count = self.image.shape[0] * self.image.shape[1]
            grayscaled_img = convert_to_grayscale(self.image)
            self.bw_img = convert_to_binary(grayscaled_img)
            self.white_pixel_count = self.calculate_white_pixels(self.bw_img)

            # Update labels
            self.total_pixel_label.setText(f"Total Pixels: {self.total_pixel_count}")
            self.white_pixel_label.setText(f"White Pixels: {self.white_pixel_count}")

            # Update the image display with the new image
            self.update_image()

    def update_image(self):
        # Clear the image for the new draw
        image_copy = self.image.copy()

        # Detect keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image_copy, None)

        # Convert keypoints to NumPy array for clustering
        keypoints_np = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # Get the current value of eps
        self.eps_value = self.eps_slider.value()

        # Update the eps value label
        self.eps_value_label.setText(f"eps: {self.eps_value}")

        # Perform DBSCAN clustering
        db = DBSCAN(eps=float(self.eps_value), min_samples=5).fit(keypoints_np)
        labels = db.labels_

        # Store the ratios for standard deviation calculation
        ratios = []

        # Count the number of bounding boxes, and store coordinates
        bounding_box_count = 0
        bbox_coordinates = []

        # Initialize the total area of bounding boxes
        total_area = 0

        # Draw bounding boxes for each cluster
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # Get the keypoints for this cluster
            cluster_keypoints = keypoints_np[labels == label]

            # Calculate bounding box
            x_min = int(np.min(cluster_keypoints[:, 0]))
            y_min = int(np.min(cluster_keypoints[:, 1]))
            x_max = int(np.max(cluster_keypoints[:, 0]))
            y_max = int(np.max(cluster_keypoints[:, 1]))

            # store bbox coordinates for calculating divisions
            bbox_coordinates.append((x_min, y_min, x_max, y_max))

            # Draw bounding box on the image if checkbox is checked
            if self.show_bounding_boxes_checkbox.isChecked():
                cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Calculate non-white and white pixel ratio from bw image
            mask = self.bw_img[y_min:y_max, x_min:x_max]
            non_white_pixels = np.sum((mask != 255))
            white_pixels = np.sum((mask == 255))

            # Avoid division by zero
            if white_pixels == 0:
                ratio = float('inf')  # Handle the case where there are no white pixels
            else:
                ratio = non_white_pixels / white_pixels

            # Calculate the area of the bounding box
            area = (x_max - x_min) * (y_max - y_min)

            # Add the area to the total area
            total_area += area

            # Store the ratio for standard deviation calculation
            ratios.append(ratio)

            # Display the ratio and area above the bounding box if checkbox is checked
            if self.show_ratio_area_labels_checkbox.isChecked():
                cv2.putText(image_copy, f'Ratio: {ratio:.2f}, Area: {area}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Increment the bounding box count
            bounding_box_count += 1

        # Update the bounding box count label
        self.bbox_count_label.setText(f"# bboxes: {bounding_box_count}")

        # Update the total area label
        self.total_area_label.setText(f"Total Area of bboxes: {total_area}")

        # Calculate and display the standard deviation of ratios
        if len(ratios) > 1:
            std_dev = np.std(ratios)
            self.std_label.setText(f"ratio(NWpx/Wpx)_std : {std_dev:.2f}")
        else:
            std_dev = 0
            self.std_label.setText("ratio(NWpx/Wpx)_std: N/A")

        # Calculate the bounding boxes / standard deviation ratio
        if std_dev > 0:
            bbox_std_ratio = bounding_box_count / std_dev
            self.bbox_std_ratio_label.setText(f"# bboxes / ratio_std: {bbox_std_ratio:.2f}")
        else:
            self.bbox_std_ratio_label.setText("# bboxes / ratio_std: N/A")

        # Calculate the new value to be displayed
        if self.white_pixel_count > 0:
            ratio_inverse_bbox_to_white_px = (self.total_pixel_count - total_area) / self.white_pixel_count
        else:
            ratio_inverse_bbox_to_white_px = float('inf')  # Handle the case where there are no white pixels

        # Update the calculated value label
        self.calculated_value_label.setText(f"(total_px_count - total_bbox_area) / white_px_count: {ratio_inverse_bbox_to_white_px:.2f}")

        # Eq maximizer, maximize this value to find ideal eps value
        if std_dev > 0:
            eq_max_value = bbox_std_ratio ** (ratio_inverse_bbox_to_white_px ** 3) # created this equation through trial and error
            # Update the calculated value label
            self.eq_maximizer.setText(f"EQ Maximizer: {eq_max_value:.2f}")
        else:
            self.eq_maximizer.setText(f"EQ Maximizer: N/A")


        # Generate divisions and draw if checked
        if self.show_divisions_checkbox.isChecked():
            image_copy, hori_lines, vert_lines = draw_dashed_lines_between_boxes(image_copy, bbox_coordinates, show_boxes=False,
                                                           alignment_tolerance=15)
            num_dashed_lines = len(hori_lines) + len(vert_lines)
            if num_dashed_lines > 0:
                self.num_divisions.setText(f"# Divisions: {num_dashed_lines}")
            else:
                self.num_divisions.setText(f"# Divisions: 0")

        # Convert the image to RGB format
        if self.show_keypoints_checkbox.isChecked():
            # Draw the keypoints
            image_with_keypoints = cv2.drawKeypoints(image_copy, keypoints, None, color=(0, 0, 255),
                                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            image_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

        # Convert to QImage for display in PySide
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Set the image in the label
        self.image_label.setPixmap(QPixmap.fromImage(q_img))


# Main entry point
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageClusterApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
