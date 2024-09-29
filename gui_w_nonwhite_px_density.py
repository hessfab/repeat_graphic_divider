import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from PySide2 import QtWidgets, QtGui, QtCore
from PySide2.QtGui import QImage, QPixmap
import sys

from test_data import convert_to_grayscale, convert_to_binary


class ImageClusterApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Load the image
        self.image_path = 'test_images/adi_x8.png'  # Replace with your image path
        self.image = cv2.imread(self.image_path)

        # Calculate total and white pixels
        self.total_pixel_count = self.image.shape[0] * self.image.shape[1]
        grayscaled_img = convert_to_grayscale(self.image)
        bw_img = convert_to_binary(grayscaled_img)
        print(self.image.shape)
        self.white_pixel_count = self.calculate_white_pixels(bw_img)

        # Initialize UI
        self.init_ui()

        # Initialize ORB detector
        self.orb = cv2.ORB_create()

        # Initial display
        self.eps_value = 30
        self.update_image()

    def init_ui(self):
        # Create layout
        layout = QtWidgets.QVBoxLayout()

        # Create a label to display the total number of pixels
        self.total_pixel_label = QtWidgets.QLabel(self)
        self.total_pixel_label.setText(f"Total Pixels: {self.total_pixel_count}")
        layout.addWidget(self.total_pixel_label)

        # Create a label to display the number of white pixels
        self.white_pixel_label = QtWidgets.QLabel(self)
        self.white_pixel_label.setText(f"White Pixels: {self.white_pixel_count}")
        layout.addWidget(self.white_pixel_label)

        # Create a label to display the total area of bounding boxes
        self.total_area_label = QtWidgets.QLabel(self)
        self.total_area_label.setText("Total Area of Bounding Boxes: N/A")
        layout.addWidget(self.total_area_label)

        # Create a label to display the calculated value
        self.calculated_value_label = QtWidgets.QLabel(self)
        self.calculated_value_label.setText("Calculated Value: N/A")
        layout.addWidget(self.calculated_value_label)

        # Create a label to display the number of bounding boxes
        self.bbox_count_label = QtWidgets.QLabel(self)
        self.bbox_count_label.setText("Number of Bounding Boxes: N/A")
        layout.addWidget(self.bbox_count_label)

        # Create a label to display the standard deviation
        self.std_label = QtWidgets.QLabel(self)
        self.std_label.setText("Standard Deviation: N/A")
        layout.addWidget(self.std_label)

        # Create a label to display bounding boxes / standard deviation
        self.bbox_std_ratio_label = QtWidgets.QLabel(self)
        self.bbox_std_ratio_label.setText("Bounding Boxes / Standard Deviation: N/A")
        layout.addWidget(self.bbox_std_ratio_label)

        # Create a label to display the calculated value
        self.eq_maximizer = QtWidgets.QLabel(self)
        self.eq_maximizer.setText("EQ Maximizer: N/A")
        layout.addWidget(self.eq_maximizer)

        # Create checkboxes for showing/hiding features
        self.show_keypoints_checkbox = QtWidgets.QCheckBox("Show Keypoints", self)
        self.show_keypoints_checkbox.setChecked(True)  # Default checked
        layout.addWidget(self.show_keypoints_checkbox)
        self.show_keypoints_checkbox.stateChanged.connect(self.update_image)

        self.show_bounding_boxes_checkbox = QtWidgets.QCheckBox("Show Bounding Boxes", self)
        self.show_bounding_boxes_checkbox.setChecked(True)  # Default checked
        layout.addWidget(self.show_bounding_boxes_checkbox)
        self.show_bounding_boxes_checkbox.stateChanged.connect(self.update_image)

        self.show_ratio_area_labels_checkbox = QtWidgets.QCheckBox("Show Ratio/Area Labels", self)
        self.show_ratio_area_labels_checkbox.setChecked(True)  # Default checked
        layout.addWidget(self.show_ratio_area_labels_checkbox)
        self.show_ratio_area_labels_checkbox.stateChanged.connect(self.update_image)

        # Create a horizontal layout for the slider and its label
        slider_layout = QtWidgets.QHBoxLayout()

        # Create a slider for adjusting the eps value
        self.eps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.eps_slider.setRange(1, 200)
        self.eps_slider.setValue(30)
        self.eps_slider.valueChanged.connect(self.update_image)
        slider_layout.addWidget(self.eps_slider)

        # Create a label to display the current eps value next to the slider
        self.eps_value_label = QtWidgets.QLabel(self)
        self.eps_value_label.setText(f"eps: {self.eps_slider.value()}")
        slider_layout.addWidget(self.eps_value_label)

        layout.addLayout(slider_layout)

        # Create a label to display the image
        self.image_label = QtWidgets.QLabel(self)
        layout.addWidget(self.image_label)

        self.setLayout(layout)
        self.setWindowTitle("DBSCAN Clustering with ORB")

    def calculate_white_pixels(self, image):
        # Count the number of white pixels (255, 255, 255)

        # for rgb image
        # white_pixels = np.sum(np.all(image == [255, 255, 255], axis=-1))

        # for bw image
        white_pixels = np.sum(image == 255)
        return white_pixels

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

        # Count the number of bounding boxes
        bounding_box_count = 0

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

            # Draw bounding box on the image if checkbox is checked
            if self.show_bounding_boxes_checkbox.isChecked():
                cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Calculate non-white and white pixel ratio
            mask = image_copy[y_min:y_max, x_min:x_max]
            non_white_pixels = np.sum((mask != [255, 255, 255]).all(axis=2))
            white_pixels = np.sum((mask == [255, 255, 255]).all(axis=2))

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
        self.bbox_count_label.setText(f"Number of Bounding Boxes: {bounding_box_count}")

        # Update the total area label
        self.total_area_label.setText(f"Total Area of Bounding Boxes: {total_area}")

        # Calculate and display the standard deviation of ratios
        if len(ratios) > 1:
            std_dev = np.std(ratios)
            self.std_label.setText(f"Standard Deviation: {std_dev:.2f}")
        else:
            std_dev = 0
            self.std_label.setText("Standard Deviation: N/A")

        # Calculate the bounding boxes / standard deviation ratio
        if std_dev > 0:
            bbox_std_ratio = bounding_box_count / std_dev
            self.bbox_std_ratio_label.setText(f"# Bounding Boxes / Standard Deviation: {bbox_std_ratio:.2f}")
        else:
            self.bbox_std_ratio_label.setText("# Bounding Boxes / Standard Deviation: N/A")

        # Calculate the new value to be displayed
        if self.white_pixel_count > 0:
            ratio_inverse_bbox_to_white_px = (self.total_pixel_count - total_area) / self.white_pixel_count
        else:
            ratio_inverse_bbox_to_white_px = float('inf')  # Handle the case where there are no white pixels

        # Update the calculated value label
        self.calculated_value_label.setText(f"(total_pixel_count - total_bbox_area) / white_pixel_count: {ratio_inverse_bbox_to_white_px:.2f}")

        # Eq maximizer, maximize this value to find ideal eps value
        if std_dev > 0:
            eq_max_value = bbox_std_ratio ** (ratio_inverse_bbox_to_white_px ** 3) # created this equation through trial and error
            # Update the calculated value label
            self.eq_maximizer.setText(f"EQ Maximizer: {eq_max_value:.2f}")
        else:
            self.eq_maximizer.setText(f"EQ Maximizer: N/A")


        # Draw the keypoints
        image_with_keypoints = cv2.drawKeypoints(image_copy, keypoints, None, color=(0, 0, 255),
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Convert the image to RGB format
        if self.show_keypoints_checkbox.isChecked():
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
