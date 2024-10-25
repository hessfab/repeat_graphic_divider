import os

import cv2
import numpy as np
from PySide2.QtWidgets import QFileDialog, QWidget, QSizePolicy
from sklearn.cluster import DBSCAN
from PySide2 import QtWidgets, QtCore
from PySide2.QtGui import QImage, QPixmap, QIcon
from PySide2.QtCore import Qt, QTimer, QThread, QObject, Signal, QSize
import sys

import test_data
from cutter import draw_dashed_lines_between_boxes
from test_data import convert_to_grayscale, convert_to_binary, add_gaussian_noise, list_files_w_ext


class TDataWorker(QObject):
    """
    Worker class responsible for generating test data in a background thread.

    Attributes:
        finished (Signal): Signal emitted when the data generation task is complete.

    Methods:
        __init__(self, tdata_size: int) -> None
            Initializes the worker with the specified size of test data to generate.

        run(self) -> None
            Executes the long-running task of generating test data and emits a signal upon completion.

    Notes:
        - This class is designed to be used with QThread to perform tasks in the background without freezing the GUI.
        - The `run` method should be connected to the thread's `started` signal to begin execution.
        - The `finished` signal should be connected to a slot that handles cleanup or further processing after data generation is complete.
    """

    finished = Signal()

    def __init__(self, tdata_size):
        super().__init__()
        self.tdata_size = tdata_size

    def run(self):
        test_data.generate_test_data(self.tdata_size)
        self.finished.emit()


class ImageClusterApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # for demo
        self.demo_running = False
        self.demo_index = 0

        # Initial display
        self.eps_value = 50

        # Load the image
        self.image_path = './test_images/test_photo_1.jpg'
        self.image_fname = os.path.basename(self.image_path)
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
        self.orb = cv2.ORB_create() #TODO create slider for nfeatures
        self.orb.setMaxFeatures(2000)


        self.update_image()

    def init_ui(self):
        # Create a horizontal layout for the image and controls
        main_layout = QtWidgets.QHBoxLayout()

        # Create a label to display the image
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        # Controls layout
        controls_layout = QtWidgets.QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignTop)
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setFixedWidth(400)
        main_layout.addWidget(controls_widget)

        # Button to open the file explorer and load a new image
        self.open_button = QtWidgets.QPushButton("Open Image")
        self.open_button.clicked.connect(self.open_image_w_dialog)
        controls_layout.addWidget(self.open_button)

        self.filename_label = QtWidgets.QLabel(f"Filename: {self.image_fname}")
        controls_layout.addWidget(self.filename_label)

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
        self.eq_maximizer_label = QtWidgets.QLabel(self)
        self.eq_maximizer_label.setText("EQ Maximizer: N/A")
        controls_layout.addWidget(self.eq_maximizer_label)

        # Create a label to display the calculated value
        self.num_divisions_label = QtWidgets.QLabel(self)
        self.num_divisions_label.setText("# Divisions: N/A")
        controls_layout.addWidget(self.num_divisions_label)

        # Create checkboxes for showing/hiding features
        self.add_noise_checkbox = QtWidgets.QCheckBox("Add Noise", self)
        self.add_noise_checkbox.setChecked(False)  # Default not checked
        controls_layout.addWidget(self.add_noise_checkbox)
        self.add_noise_checkbox.stateChanged.connect(self.update_image)

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
        self.eps_max = 100
        self.eps_slider.setRange(1, self.eps_max)
        self.eps_slider.setValue(self.eps_value)
        self.eps_slider.valueChanged.connect(self.update_image)
        slider_layout.addWidget(self.eps_slider)

        # Create a label to display the current eps value next to the slider
        self.eps_value_label = QtWidgets.QLabel(self)
        self.eps_value_label.setText(f"eps: {self.eps_slider.value()}")
        slider_layout.addWidget(self.eps_value_label)

        controls_layout.addLayout(slider_layout)

        # test data widgets
        tdata_hbox = QtWidgets.QHBoxLayout()

        # Create a label for tdata size
        self.tdata_size_label = QtWidgets.QLabel("Select Size:")
        tdata_hbox.addWidget(self.tdata_size_label)

        # Create a combo box with numbers from 10 to 100 in steps of 10
        self.tdata_combo_box = QtWidgets.QComboBox()
        self.tdata_combo_box.addItems([str(i) for i in range(10, 110, 10)])
        tdata_hbox.addWidget(self.tdata_combo_box)

        # Create a button generate test data
        self.gen_data_btn = QtWidgets.QPushButton("Generate Test Data")
        tdata_hbox.addWidget(self.gen_data_btn)
        self.gen_data_btn.clicked.connect(self.generate_test_data)

        controls_layout.addLayout(tdata_hbox)

        self.spacer = QWidget()
        self.spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        controls_layout.addWidget(self.spacer)

        # scroll image widgets
        scroll_hbox = QtWidgets.QHBoxLayout()

        # Create a button on the left with an arrow pointing to the left icon
        left_arrow_icon = QIcon("./assets/icons/icons8-left-arrow-72.png")
        self.left_button = QtWidgets.QPushButton()
        self.left_button.setIcon(left_arrow_icon)
        self.left_button.setIconSize(QSize(36, 36))
        self.left_button.setDisabled(True)
        scroll_hbox.addWidget(self.left_button)

        # Connect the left button's clicked signal to a slot
        self.left_button.clicked.connect(self.previous_image)

        # Create a button on the right with an arrow pointing to the right icon
        right_arrow_icon = QIcon("./assets/icons/icons8-right-arrow-72.png")
        self.right_button = QtWidgets.QPushButton()
        self.right_button.setIcon(right_arrow_icon)
        self.right_button.setIconSize(QSize(36, 36))
        self.right_button.setDisabled(True)
        scroll_hbox.addWidget(self.right_button)

        # Connect the right button's clicked signal to a slot
        self.right_button.clicked.connect(self.next_image)

        controls_layout.addLayout(scroll_hbox)

        # Button to run demo
        self.run_demo_button = QtWidgets.QPushButton("Run Demo")
        self.run_demo_button.clicked.connect(self.run_demo)
        controls_layout.addWidget(self.run_demo_button)

        self.msg_label = QtWidgets.QLabel("")
        controls_layout.addWidget(self.msg_label)

        self.setLayout(main_layout)
        self.setWindowTitle("Repeat Graphic Divider - Demo")

    def next_image(self):
        self.demo_index = (self.demo_index + 1) % len(self.demo_images)
        self.open_image(self.demo_images[self.demo_index])

    def previous_image(self):
        self.demo_index = (self.demo_index - 1) % len(self.demo_images)
        self.open_image(self.demo_images[self.demo_index])

    def generate_test_data(self):
        tdata_size = int(self.tdata_combo_box.currentText())
        self.msg_label.setText("Generating test data...")

        self.thread = QThread()
        self.worker = TDataWorker(tdata_size)

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        # self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.tdata_gen_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        # self.msg_label.setText("Test data generation complete!")

    def tdata_gen_finished(self):
        self.msg_label.setText("Test data generation complete!")

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
        # Only downscale, don't upscale
        if aspect_ratio < 1.0:
            self.image = cv2.resize(self.image, (new_width, new_height))

    def open_image(self, file_path):
        if file_path:
            # Load the new image
            self.image_path = file_path
            self.image_fname = os.path.basename(self.image_path)
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
            self.filename_label.setText(f"Filename: {self.image_fname}")

            # Update the image display with the new image
            self.update_image()

    def open_image_w_dialog(self):
        # Open a file dialog to select the image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)")
        self.open_image(file_path)

    def change_eps_update_image(self):
        if self.demo_index == 0:
            self.open_image(self.demo_images[self.demo_index])
        if self.eps_value >= self.eps_max:
            # max eps reached, reset
            self.eps_value = 0
            self.eps_slider.setValue(self.eps_value)
            # move to next image
            self.demo_index += 1
            if self.demo_index == len(self.demo_images):
                self.timer.stop()
                self.demo_finished()
            else:
                self.open_image(self.demo_images[self.demo_index])
        else:
            # increase eps value and update
            self.eps_value += 1
            self.eps_slider.setValue(self.eps_value)
            self.update_image()

    def demo_finished(self):
        # print("Demo completed.")
        self.demo_index = 0
        self.open_image(self.demo_images[self.demo_index])
        self.demo_running = False
        self.run_demo_button.setText("Run Demo")
        self.msg_label.setText("Demo Finished!")
        self.left_button.setDisabled(False)
        self.right_button.setDisabled(False)

    def run_demo(self):
        # print("running demo")
        self.demo_running = not self.demo_running
        if self.demo_running:
            self.demo_images = list_files_w_ext(test_data.output_folder, "png")
            if len(self.demo_images) == 0:
                QtWidgets.QMessageBox.critical(self, "Error", "No test data found in ./test_data/output_images to run demo on. Please generate test data first.")
                self.demo_running = False
            else:

                self.eps_value = 0
                # change button text
                self.run_demo_button.setText("Stop Demo")
                # Connect QTimer to trigger image updates
                self.timer = QTimer()
                self.timer.timeout.connect(self.change_eps_update_image)
                self.timer.start(20)  # 20 milliseconds

                self.msg_label.setText("Running Demo...")
                self.left_button.setDisabled(True)
                self.right_button.setDisabled(True)

        else:
            # demo stopped
            self.timer.stop()
            # change button text
            self.run_demo_button.setText("Run Demo")
            self.msg_label.setText("Demo stopped!")
            self.left_button.setDisabled(False)
            self.right_button.setDisabled(False)

        # print(self.demo_running)

    def update_image(self):
        # Clear the image for the new draw
        image_copy = self.image.copy()

        # add noise if checked
        if self.add_noise_checkbox.isChecked():
            image_copy = add_gaussian_noise(image_copy)

        # Detect keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image_copy, None)

        # Convert keypoints to NumPy array for clustering
        keypoints_np = np.array([kp.pt for kp in keypoints], dtype=np.float32)

        # Get the current value of eps
        self.eps_value = self.eps_slider.value()

        # Update the eps value label
        self.eps_value_label.setText(f"eps: {self.eps_value}")

        # Perform DBSCAN clustering
        db = DBSCAN(eps=float(self.eps_value), min_samples=10).fit(keypoints_np) #TODO slider for min samples
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
            self.eq_maximizer_label.setText(f"EQ Maximizer: {eq_max_value:.2f}")
        else:
            self.eq_maximizer_label.setText(f"EQ Maximizer: N/A")


        # Generate divisions and draw if checked
        if self.show_divisions_checkbox.isChecked():
            image_copy, hori_lines, vert_lines = draw_dashed_lines_between_boxes(image_copy, bbox_coordinates, show_boxes=False,
                                                           alignment_tolerance=15)
            num_dashed_lines = len(hori_lines) + len(vert_lines)
            if num_dashed_lines > 0:
                self.num_divisions_label.setText(f"# Divisions: {num_dashed_lines}")
            else:
                self.num_divisions_label.setText(f"# Divisions: 0")

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

        original_pixmap = QPixmap.fromImage(q_img)

        # Set the maximum width and height
        max_width = 800
        max_height = 800

        # Get original dimensions
        original_width = original_pixmap.width()
        original_height = original_pixmap.height()

        # Calculate the scaling factor while respecting both max width and height
        scale_factor = min(max_width / original_width, max_height / original_height)

        # Scale the pixmap while maintaining the aspect ratio
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        scaled_pixmap = original_pixmap.scaled(new_width, new_height)

        # Set the image in the label
        self.image_label.setPixmap(scaled_pixmap)

        # resize window to fit new content
        # self.adjustSize()


# Main entry point
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageClusterApp()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())
