import sys
import os
import time
import numpy as np
import cv2
from screeninfo import get_monitors
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QFont
#from pypylon import pylon
import utils


class ConfirmationDialog(QWidget):
	parameter_signal = pyqtSignal(bool)


	def __init__(self):
		super().__init__()
		self.setWindowTitle("Confirm background")
		font = QFont()
		font.setPointSize(12)

		layout_main = QVBoxLayout()
		layout_main.setContentsMargins(20, 20, 20, 20)

		self.label = QLabel('Are you happy with the background image (the animal should not be visible) and want to proceed to the experiment?')
		self.label.setWordWrap(True)
		self.label.setAlignment(Qt.AlignCenter)
		self.label.setFont(font)
		layout_main.addWidget(self.label)

		layout = QHBoxLayout()
		self.button_yes = QPushButton('Yes, proceed')
		self.button_yes.setFont(font)
		self.button_yes.clicked.connect(self.emit_true)
		layout.addWidget(self.button_yes)
		self.button_no = QPushButton('No, repeat')
		self.button_no.setFont(font)
		self.button_no.clicked.connect(self.emit_false)
		layout.addWidget(self.button_no)
		layout_main.addLayout(layout)

		self.setLayout(layout_main)
		self.setMinimumWidth(280)
		self.adjustSize()

		_, _, screen_width, screen_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(screen_width * 3 // 4 - width // 2, screen_height // 2 - height // 2)


	def emit_true(self, _):
		self.parameter_signal.emit(True)

	def emit_false(self, _):
		self.parameter_signal.emit(False)

	def closeEvent(self, event):
		QApplication.exit(0)


class SnapshotDialog(QWidget):
	parameter_signal = pyqtSignal()


	def __init__(self):
		super().__init__()
		self.setWindowTitle("Extract background")
		font = QFont()
		font.setPointSize(12)

		layout_main = QVBoxLayout()
		layout_main.setContentsMargins(20, 20, 20, 20)

		self.label_info = QLabel('Take three snapshots where\nthe animal is at three distinct\nnon-overlapping positions.')
		self.label_info.setAlignment(Qt.AlignCenter)
		self.label_info.setFont(font)

		self.button = QPushButton('Snap!')
		self.button.setFont(font)
		self.button.clicked.connect(self.emit_snap)

		self.label_count = QLabel('Snaps left: 3')
		self.label_count.setAlignment(Qt.AlignCenter)
		self.label_count.setFont(font)

		layout_main.addWidget(self.label_info)
		layout_main.addWidget(self.button)
		layout_main.addWidget(self.label_count)

		self.setLayout(layout_main)
		self.setMinimumWidth(275)
		self.adjustSize()

		_, _, screen_width, screen_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(screen_width * 3 // 4 - width // 2, screen_height // 2 - height // 2)


	def reset_counter(self):
		self.label_count.setText('Snaps left: 3')
		

	def emit_snap(self, _):
		self.label_count.setText(f'Snaps left: {int(self.label_count.text()[-1]) - 1}')
		self.parameter_signal.emit()

	def closeEvent(self, event):
		QApplication.exit(0)


def compute_background(images):
	# iterative algorithm version
	'''image_bg = np.zeros(images[0].shape, dtype=np.uint8)
	for i in range(images[0].shape[0]):
		for j in range(images[0].shape[1]):
			# find dots with smallest distance
			pixels = (images[0][i, j], images[1][i, j], images[2][i, j])
			pixel_pairs = ((pixels[0], pixels[1]), (pixels[0], pixels[2]), (pixels[1], pixels[2]))
			distances = list(map(lambda pair: np.uint8(abs(np.intp(pair[0]) - np.intp(pair[1]))), pixel_pairs))
			min_dist_pixels = pixel_pairs[np.argmin(np.array(distances))]
			image_bg[i, j] = np.mean(min_dist_pixels)'''
	
	# vectorized numpy black magic version
	image_tensor = np.stack(images, axis=-1)  # shape: (H, W, 3)
	distance_tensor = np.stack([
		np.abs(image_tensor[..., 0].astype(np.int16) - image_tensor[..., 1].astype(np.int16)).astype(np.uint8),
		np.abs(image_tensor[..., 0].astype(np.int16) - image_tensor[..., 2].astype(np.int16)).astype(np.uint8),
		np.abs(image_tensor[..., 1].astype(np.int16) - image_tensor[..., 2].astype(np.int16)).astype(np.uint8)
	], axis=-1) # shape: (H, W, 3)
	min_dist_indices = np.argmin(distance_tensor, axis=-1)  # shape: (H, W)
	pairs = np.array([
		[image_tensor[..., 0], image_tensor[..., 1]],
		[image_tensor[..., 0], image_tensor[..., 2]],
		[image_tensor[..., 1], image_tensor[..., 2]],
	])  # shape: (3, 2, H, W)
	min_dist_pairs = np.take_along_axis(pairs, min_dist_indices[None, None, :, :], axis=0)  # shape: (1, 2, H, W)
	image_bg = np.mean(min_dist_pairs.astype(np.float16), axis=1).squeeze().astype(np.uint8)  # shape: (H, W)

	# return
	return image_bg


def main():
	# camera setup
	#camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
	#camera.Open()
	#if not camera.IsOpen():
		#utils.show_error('Cannot find camera.')
		#return
	#utils.setupCamera(camera)
	#image_size = (camera.Width.GetValue(), camera.Height.GetValue())
	dirpath = utils.choose_directory('Select input video directory.')
	img_names = sorted(os.listdir(dirpath), key=lambda name: int(name.split('_')[-1].split('.')[0]))
	image_size = (1544, 1544)
	i_c = 1000 #7000

	# window setup
	monitors = get_monitors()
	window_name = 'Video/Background'
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, monitors[0].height * 8 // 10, monitors[0].height * 8 // 10)
	cv2.moveWindow(window_name, monitors[0].x + monitors[0].height * 1 // 10, monitors[0].y + monitors[0].height * 1 // 10)

	# confirmation dialog setup
	satisfied = False
	interacted = False
	def update_status(satisfied_new):
		nonlocal satisfied
		satisfied = satisfied_new
		nonlocal interacted
		interacted = True
	app = QApplication(sys.argv)
	confirmation_dialog = ConfirmationDialog()
	confirmation_dialog.parameter_signal.connect(update_status)

	# snap dialog setup
	frame_counter = 0
	increased = False
	def snap():
		nonlocal increased
		increased = True
	snapshot_dialog = SnapshotDialog()
	snapshot_dialog.parameter_signal.connect(snap)

	# outer loop
	flash_image = np.ones(image_size)
	background_candidate = None
	frames = np.stack([
		np.empty(image_size, dtype=np.uint8),
		np.empty(image_size, dtype=np.uint8),
		np.empty(image_size, dtype=np.uint8),
		np.empty(image_size, dtype=np.uint8)
	], axis=0)
	while not satisfied:

		# main loop
		snapshot_dialog.show()
		#camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
		while i_c < len(img_names) and frame_counter <= 2:  # camera.IsGrabbing()
			#grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
			#if not grabResult.GrabSucceeded(): break
			#frame = cv2.flip(grabResult.Array, -1)
			frame = cv2.flip(cv2.imread(os.path.join(dirpath, img_names[i_c])), -1)
			#grabResult.Release()
			if len(frame.shape) != 2:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frames[0] = cv2.resize(frame, image_size)

			visual_frame = np.mean(frames[:frame_counter + 1], axis=0).astype(np.uint8)
			cv2.imshow(window_name, visual_frame)
			cv2.waitKey(1)

			if increased:
				frame_counter = frame_counter + 1
				frames[frame_counter] = frames[0].copy()
				increased = False

				cv2.imshow(window_name, flash_image)
				cv2.waitKey(16)

			i_c += 1
		#camera.StopGrabbing()
		snapshot_dialog.hide()

		background_candidate = compute_background(frames)
	
		confirmation_dialog.show()
		while not interacted:
			cv2.imshow(window_name, background_candidate)
			cv2.waitKey(1)
		interacted = False
		confirmation_dialog.hide()

		frame_counter = 0
		snapshot_dialog.reset_counter()


	cv2.destroyAllWindows()
	app.quit()

	os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
	cv2.imwrite(os.path.join(os.getcwd(), 'data', f'background.jpg'), background_candidate) # type: ignore


main()
