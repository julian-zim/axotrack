from sys import argv
from os import getcwd, makedirs
from os.path import exists, join
from numpy import empty, abs, array, full, stack, mean, argmin, take_along_axis, uint8, int16, float16
from screeninfo import get_monitors
from cv2 import namedWindow, moveWindow, resizeWindow, setWindowProperty, getWindowProperty, destroyAllWindows, WINDOW_NORMAL, WINDOW_FULLSCREEN, WND_PROP_FULLSCREEN, WND_PROP_VISIBLE
from cv2 import waitKey, imshow, imwrite, resize, flip, cvtColor, COLOR_BGR2GRAY
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QDesktopWidget, QWidget, QLabel, QSlider, QPushButton, QSpacerItem, QSizePolicy
from pypylon.pylon import InstantCamera, TlFactory, GrabStrategy_LatestImageOnly, TimeoutHandling_ThrowException
from utils import show_info, show_error, setup_camera


class ConfirmationDialog(QWidget):
	signal = pyqtSignal(bool)

	def __init__(self):
		super().__init__()

		font = QFont()
		font.setPointSize(12)

		self.label = QLabel('Are you happy with the background image (the animal should not be visible) and want to proceed to the experiment?')
		self.label.setWordWrap(True)
		self.label.setAlignment(Qt.AlignCenter)
		self.label.setFont(font)

		layout = QHBoxLayout()
		self.button_yes = QPushButton('Yes, proceed')
		self.button_yes.setFont(font)
		self.button_yes.clicked.connect(self.emit_true)
		layout.addWidget(self.button_yes)
		self.button_no = QPushButton('No, repeat')
		self.button_no.setFont(font)
		self.button_no.clicked.connect(self.emit_false)
		layout.addWidget(self.button_no)

		layout_main = QVBoxLayout()
		layout_main.setContentsMargins(20, 20, 20, 20)
		layout_main.addWidget(self.label)
		layout_main.addLayout(layout)
		self.setLayout(layout_main)

		self.setWindowTitle("Confirm background")
		self.setMinimumWidth(280)
		self.adjustSize()
		_, _, screen_width, screen_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(screen_width * 3 // 4 - width // 2, screen_height // 2 - height // 2)

	def emit_true(self, _):
		self.signal.emit(True)

	def emit_false(self, _):
		self.signal.emit(False)

	def closeEvent(self, event):
		QApplication.exit(0)


class SnapshotDialog(QWidget):
	signal_snap = pyqtSignal()
	signal_brightness = pyqtSignal(int)

	def __init__(self):
		super().__init__()

		font = QFont()
		font.setPointSize(12)

		self.label_info = QLabel('Take three snapshots where\nthe animal is at three distinct\nnon-overlapping positions.')
		self.label_info.setAlignment(Qt.AlignCenter)
		self.label_info.setFont(font)
	
		self.slider_brightness = QSlider(Qt.Horizontal)
		self.slider_brightness.setMinimum(0)
		self.slider_brightness.setMaximum(255)
		self.slider_brightness.setValue(255)
		self.slider_brightness.valueChanged.connect(self.emit_brightness)

		self.button_snap = QPushButton('Snap!')
		self.button_snap.setFont(font)
		self.button_snap.clicked.connect(self.emit_snap)

		self.label_count = QLabel('Snaps left: 3')
		self.label_count.setAlignment(Qt.AlignCenter)
		self.label_count.setFont(font)

		spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
				
		self.button_skip = QPushButton('Skip this step')
		self.button_skip.setFont(font)
		self.button_skip.clicked.connect(self.close)

		layout_main = QVBoxLayout()
		layout_main.setContentsMargins(20, 20, 20, 20)

		layout_main.addWidget(self.label_info)
		layout_main.addWidget(self.slider_brightness)
		layout_main.addWidget(self.button_snap)
		layout_main.addWidget(self.label_count)
		layout_main.addItem(spacer)
		layout_main.addWidget(self.button_skip)

		self.setLayout(layout_main)

		self.setWindowTitle("Extract background")
		self.setMinimumWidth(275)
		self.adjustSize()
		_, _, screen_width, screen_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(screen_width * 3 // 4 - width // 2, screen_height // 2 - height // 2)

	def reset(self):
		self.slider_brightness.setEnabled(True)
		self.label_count.setText('Snaps left: 3')

	def emit_brightness(self):
		self.signal_brightness.emit(self.slider_brightness.value())

	def emit_snap(self):
		self.slider_brightness.setEnabled(False)
		self.label_count.setText(f'Snaps left: {int(self.label_count.text()[-1]) - 1}')
		self.signal_snap.emit()

	def closeEvent(self, event):
		QApplication.exit(0)


def compute_background(images):
	# iterative algorithm version
	'''image_bg = zeros(images[0].shape, dtype=uint8)
	for i in range(images[0].shape[0]):
		for j in range(images[0].shape[1]):
			# find dots with smallest distance
			pixels = (images[0][i, j], images[1][i, j], images[2][i, j])
			pixel_pairs = ((pixels[0], pixels[1]), (pixels[0], pixels[2]), (pixels[1], pixels[2]))
			distances = list(map(lambda pair: uint8(abs(intp(pair[0]) - intp(pair[1]))), pixel_pairs))
			min_dist_pixels = pixel_pairs[argmin(array(distances))]
			image_bg[i, j] = mean(min_dist_pixels)'''
	
	# vectorized numpy black magic version
	image_tensor = stack(images, axis=-1)  # shape: (H, W, 3)
	distance_tensor = stack([
		abs(image_tensor[..., 0].astype(int16) - image_tensor[..., 1].astype(int16)).astype(uint8),
		abs(image_tensor[..., 0].astype(int16) - image_tensor[..., 2].astype(int16)).astype(uint8),
		abs(image_tensor[..., 1].astype(int16) - image_tensor[..., 2].astype(int16)).astype(uint8)
	], axis=-1) # shape: (H, W, 3)
	min_dist_indices = argmin(distance_tensor, axis=-1)  # shape: (H, W)
	pairs = array([
		[image_tensor[..., 0], image_tensor[..., 1]],
		[image_tensor[..., 0], image_tensor[..., 2]],
		[image_tensor[..., 1], image_tensor[..., 2]],
	])  # shape: (3, 2, H, W)
	min_dist_pairs = take_along_axis(pairs, min_dist_indices[None, None, :, :], axis=0)  # shape: (1, 2, H, W)
	image_bg = mean(min_dist_pairs.astype(float16), axis=1).squeeze().astype(uint8)  # shape: (H, W)

	return image_bg


def main():
	# camera setup
	try:
		camera = InstantCamera(TlFactory.GetInstance().CreateFirstDevice())
	except:
		show_error('Could not access the camera. Make sure all other software that accesses it is closed.')
		return
	camera.Open()
	if not camera.IsOpen():
		show_error('Cannot find camera.')
		return
	setup_camera(camera)

	# window setup
	monitors = get_monitors()

	controlwindow_name = 'Camera / Background Preview'
	namedWindow(controlwindow_name, WINDOW_NORMAL)
	resizeWindow(controlwindow_name, monitors[0].height * 8 // 10, monitors[0].height * 8 // 10)
	moveWindow(controlwindow_name, monitors[0].x + monitors[0].height * 1 // 10, monitors[0].y + monitors[0].height * 1 // 10)

	if len(monitors) > 1:
		projwindow_name = 'Projection Window'
		namedWindow(projwindow_name, WINDOW_NORMAL)
		moveWindow(projwindow_name, monitors[1].x, monitors[1].y)
		resizeWindow(projwindow_name, monitors[1].width, monitors[1].height)
		setWindowProperty(projwindow_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)
		image_backlight = full((monitors[1].width, monitors[1].height), 255, dtype=uint8)
	else:
		show_info('No second screen detected!')

	# snap dialog setup
	def update_brightness(brightness_new):
		if len(monitors) > 1:
			nonlocal image_backlight
			image_backlight.fill(brightness_new)
	increased = False
	def update_snap():
		nonlocal increased
		increased = True
	app = QApplication(argv)
	snapshot_dialog = SnapshotDialog()
	snapshot_dialog.signal_brightness.connect(update_brightness)
	snapshot_dialog.signal_snap.connect(update_snap)

	# confirmation dialog setup
	satisfied = False
	interacted = False
	def update_status(satisfied_new):
		nonlocal satisfied, interacted
		satisfied = satisfied_new
		interacted = True
	confirmation_dialog = ConfirmationDialog()
	confirmation_dialog.signal.connect(update_status)

	# outer loop
	image_square_shape = (max(camera.Width.GetValue(), camera.Height.GetValue()), max(camera.Width.GetValue(), camera.Height.GetValue()))
	image_flash = full(image_square_shape, 255, dtype=uint8)
	image_background = empty(())
	images = stack([
		empty(image_square_shape, dtype=uint8),  # active frame
		empty(image_square_shape, dtype=uint8),  # snapped frame #1
		empty(image_square_shape, dtype=uint8),  # snapped frame #2
		empty(image_square_shape, dtype=uint8)   # snapped frame #3
	], axis=0)
	image_counter = 0
	broke = False
	while not satisfied and not broke:

		# main loop
		snapshot_dialog.show()
		camera.StartGrabbing(GrabStrategy_LatestImageOnly)
		while camera.IsGrabbing() and image_counter <= 2:
			if not snapshot_dialog.isVisible() or not getWindowProperty(controlwindow_name, WND_PROP_VISIBLE):
				broke = True
				break

			grabResult = camera.RetrieveResult(1000, TimeoutHandling_ThrowException)
			if not grabResult.GrabSucceeded(): break
			frame = flip(grabResult.Array, -1)
			grabResult.Release()
			if len(frame.shape) != 2:
				frame = cvtColor(frame, COLOR_BGR2GRAY)

			images[0] = resize(frame, image_square_shape)
			image_visual = mean(images[:image_counter + 1], axis=0).astype(uint8)

			imshow(controlwindow_name, image_visual)
			if len(monitors) > 1: imshow(projwindow_name, image_backlight)
			waitKey(1)

			if increased:
				image_counter = image_counter + 1
				images[image_counter] = images[0].copy()
				increased = False

				imshow(controlwindow_name, image_flash)
				waitKey(16)

		# stop
		camera.StopGrabbing()
		if broke: break
		snapshot_dialog.hide()

		# assemble background
		image_background = compute_background(images)
	
		# secondary loop
		confirmation_dialog.show()
		while not interacted:
			if not confirmation_dialog.isVisible() or not getWindowProperty(controlwindow_name, WND_PROP_VISIBLE):
				broke = True
				break
	
			imshow(controlwindow_name, image_background)
			waitKey(1)
		if broke: break

		# reset
		interacted = False
		confirmation_dialog.hide()

		image_counter = 0
		snapshot_dialog.reset()

	# quit
	camera.Close()
	destroyAllWindows()
	confirmation_dialog.close()
	snapshot_dialog.close()
	app.quit()

	# save
	if broke:
		if exists(join(getcwd(), 'data', 'background.jpg')):
			background_info = f'\n\nOld background image was found at\n"{getcwd()}\\data\\background.jpg"\nand will be used.'
		else:
			background_info = '\n\nNote that no old background image was found.'

		if exists(join(getcwd(), 'data', 'backlight.txt')):
			backlight_info = f'\n\nOld backlight file was found at\n"{getcwd()}\\data\\backlight.txt"\nand will be used.'
		else:
			backlight_info = '\n\nNote that no old backlight file was found.'
		show_info(f'Capture canceled.{background_info}{backlight_info}')

		return
	
	out_dirpath = join(getcwd(), 'data')
	makedirs(out_dirpath, exist_ok=True)
	imwrite(join(out_dirpath, 'background.jpg'), image_background)
	if len(monitors) > 1:
		with open(join(out_dirpath, 'backlight.txt'), 'w') as file:
			file.write(str(image_backlight[0, 0]))
		show_info(f'Background data saved to\n\n"{getcwd()}\\data\\"\n\nas "background.jpg" and "backlight.txt".')
	else :
		show_info(f'Background image saved to\n\n"{getcwd()}\\data\\background.jpg".')


if __name__ == '__main__':
	print('PROGRAM STARTED')
	main()
