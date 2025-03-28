import math
import random
import sys
import os
import time
import numpy as np
from screeninfo import get_monitors
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QDesktopWidget
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from joblib import dump, load
# placeholder
import utils


# constants
precision = 100
max_fps = 30
trackwindow_screen_ratio = 1.5
diff_window_screen_ratio = 2.5

default_titles = [
	'Max Tracker Width',
	'Max Tracker Height',
	'Dot Active',  # use checkbox
	'Periphery',  # use checkbox
	'Lateral Distance',
	'Medial Distance',
	'Dot Color',  # user sliders
	'Dot Size',
	'Jitter Speed',  # use slider
	'Jitter Magnitude',
]
default_values = [
	'50',
	'50',
	'0',
	'-1',
	'30',
	'10',
	'0,0,0',
	'2',
	'50',
	'10'
]

# global variables
track_window = (0, 0, default_values[0], default_values[1])


class ParameterControl(QWidget):
	parameter_signal = pyqtSignal(*[str for _ in range(len(default_titles))])

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Control Panel")
		self.resize(250, 198)  # tweak

		layout = QVBoxLayout()
		self.labels = [QLabel(default_title) for default_title in default_titles]
		for label in self.labels:
			label.setFixedSize(100, 20)  # tweak
		self.edits = [QLineEdit(default_value) for default_value in default_values]
		assert(len(self.labels) == len(self.edits))
		for i in range(len(self.edits)):
			sub_layout = QHBoxLayout()
			sub_layout.addWidget(self.labels[i])
			sub_layout.addWidget(self.edits[i])
			layout.addLayout(sub_layout)
		self.setLayout(layout)

		_, _, projector_width, projector_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(projector_width * 3 // 4 - width // 2, projector_height // 4 - height // 2) # type: ignore

		for i in range(len(self.edits)):
			self.edits[i].editingFinished.connect(self.emit_values)

	def emit_values(self):
		# for tracker size
		for i in [0, 1]:
			try:
				ival = int(self.edits[i].text())
				assert(ival > 0)
			except (ValueError, AssertionError):
				self.edits[i].setText(default_values[i])
		# for active
		i = 2
		if self.edits[i].text().upper() not in ['0', '1']:
			self.edits[i].setText(default_values[i])
		# for periphery
		i = 3
		if self.edits[i].text().upper() not in ['-1', '1']:
			self.edits[i].setText(default_values[i])
		# for distance
		for i in [4, 5]:
			try:
				float(self.edits[i].text())
			except ValueError:
				self.edits[i].setText(default_values[i])
		# for color
		i = 6
		vals = self.edits[i].text().split(',')
		if len(vals) != 3:
			self.edits[i].setText(default_values[i])
		else:
			try:
				fvals = [float(val) for val in vals]
				assert(all([0. <= fval <= 255. for fval in fvals]))
			except (ValueError, AssertionError):
				self.edits[i].setText(default_values[i])
		# for size, magnitude
		for i in [7, 9]:
			try:
				fval = float(self.edits[i].text())
				assert(fval >= 0.)
			except (ValueError, AssertionError):
				self.edits[i].setText(default_values[i])
		# for speed
		i = 8
		try:
			fval = float(self.edits[i].text())
			assert(0. <= fval <= 100.)
		except (ValueError, AssertionError):
			self.edits[i].setText(default_values[i])
	
		self.parameter_signal.emit(*[edit.text() for edit in self.edits])

	def closeEvent(self, event):
		QApplication.exit(0)


def mouse_callback(event, x, y, flags, param):
	global track_window

	if event == cv2.EVENT_LBUTTONDOWN:
		track_window = (x, y, track_window[2], track_window[3])


def extract_background(path, precision=50):
	if os.path.exists(os.path.join(path, 'Background.tiff')):
		print('Found background file!')
		return

	print('Extracting background...')
	imgs = [cv2.cvtColor(cv2.imread(os.path.join(path, img_name)), cv2.COLOR_BGR2GRAY) for img_name in os.listdir(path)[::round(100 / precision)]]
	img_bg = np.median(np.array(imgs).astype(np.float16), axis=0).astype(np.uint8)
	cv2.imwrite(os.path.join(path, 'Background.tiff'), img_bg)
	print('Done!')


def rotate(image, angle):
	height, width = image.shape

	max_dim = max([width, height])
	diff_w = (max_dim - width) // 2
	diff_h = (max_dim - height) // 2
	image = np.pad(image, ((diff_h, diff_h), (diff_w, diff_w)), mode='constant', constant_values=0)

	rotation_matrix = cv2.getRotationMatrix2D((max_dim // 2, max_dim // 2), angle, 1.0)
	rotated_image = cv2.warpAffine(image, rotation_matrix, (max_dim, max_dim))

	return rotated_image


def main():
	# region VARIABLES
	max_track_window_size = (int(default_values[0]), int(default_values[1]))
	active = bool(int(default_values[2]))
	periphery = (int(default_values[3]) + 1) // 2
	distance_x = float(default_values[4])
	distance_y = float(default_values[5])
	color = (float(default_values[6].split(',')[2]) / 255., float(default_values[6].split(',')[1]) / 255., float(default_values[6].split(',')[0]) / 255.)
	size = float(default_values[7])
	jitter_speed = float(default_values[8])
	jitter_magnitude = round(float(default_values[9]))

	def update_values(
			tracker_width_new,
			tracker_height_new,
			active_new,
			periphery_new, 
			distance_x_new,
			distance_y_new,
			color_new,
			size_new,
			jitter_speed_new,
			jitter_magnitude_new
		):
		nonlocal max_track_window_size, active, periphery, distance_x, distance_y, color, size, jitter_speed, jitter_magnitude
		max_track_window_size = (int(tracker_width_new), int(tracker_height_new))
		active = bool(int(active_new))
		periphery = (int(periphery_new) + 1) // 2
		distance_x = float(distance_x_new)
		distance_y = float(distance_y_new)
		color = (float(color_new.split(',')[2]) / 255., float(color_new.split(',')[1]) / 255., float(color_new.split(',')[0]) / 255.)
		size = float(size_new)
		jitter_speed = float(jitter_speed_new)
		jitter_magnitude = round(float(jitter_magnitude_new))
	# endregion

	# region SETUP

	# pyqt stuff
	app = QApplication(sys.argv)
	controller = ParameterControl()
	controller.parameter_signal.connect(update_values)

	# get monitors
	monitors = get_monitors()
	# placeholder
	# placeholder
	# placeholder
	first_monitor = monitors[0]

	# get background image
	path = utils.choose_directory()
	img_names = os.listdir(path)
	if 'Background.tiff' in img_names: img_names.remove('Background.tiff')
	if 'processed' in img_names: img_names.remove('processed')
	img_names = sorted(img_names, key=lambda name: int(name.split('_')[-1].split('.')[0]))
	extract_background(path, precision)
	os.makedirs(os.path.join(path, 'processed'), exist_ok=True)
	image_bg = cv2.imread(os.path.join(path, 'Background.tiff'))
	if len(image_bg.shape) != 2:
		image_bg = cv2.cvtColor(image_bg, cv2.COLOR_BGR2GRAY)
	image_bg_width, image_bg_height = image_bg.shape[1], image_bg.shape[0]
	projector_width, projector_height = image_bg_width, image_bg_height
	projector_image_ratio = float(image_bg_height) / float(projector_height)  # should be same as image_width / projector_width

	# get zoom & shift
	delta_x, delta_y = 0, 0
	delta_z = 1.

	# setup windows
	# placeholder
	# placeholder
	# placeholder
	# placeholder

	trackwindow_name = 'Tracking View'
	trackwindow_width, trackwindow_height = round(projector_width / trackwindow_screen_ratio), round(projector_height / trackwindow_screen_ratio)
	cv2.namedWindow(trackwindow_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(trackwindow_name, trackwindow_width * 2, trackwindow_height * 2)
	cv2.moveWindow(trackwindow_name, first_monitor.x + (first_monitor.height - trackwindow_height) // 2, first_monitor.y + (first_monitor.height - trackwindow_height) // 2)

	diff_window_name = 'Difference View'
	diff_window_width, diff_window_height = int(round(projector_width / diff_window_screen_ratio)), round(projector_height / diff_window_screen_ratio)
	cv2.namedWindow(diff_window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(diff_window_name, diff_window_width * 2, diff_window_height * 2)
	cv2.moveWindow(diff_window_name, first_monitor.width * 3 // 4 - diff_window_width // 2, first_monitor.y + first_monitor.height // 2)

	# setup tracking
	global track_window
	track_window = (image_bg_width // 2 - max_track_window_size[0] // 2, image_bg_height // 2 - max_track_window_size[1] // 2, max_track_window_size[0], max_track_window_size[1])
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

	# control panel
	controller.show()

	#endregion

	# loop
	frame_delay = round(1000. / max_fps)
	noise = (0., 0.)
	counter = 0
	i_c = 0
	while i_c < len(img_names):
		start = time.perf_counter()

		# quit
		if cv2.getWindowProperty(trackwindow_name, cv2.WND_PROP_VISIBLE) + cv2.getWindowProperty(diff_window_name, cv2.WND_PROP_VISIBLE) < 2 or not controller.isVisible():
			break

		# read camera
		frame = cv2.imread(os.path.join(path, img_names[i_c]))
		i_c += round(max_fps / 30)  # fps of recording
		image_orig = frame
		if len(image_orig.shape) != 2:
			image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)

		# image preprocessing
		projector_ratio = float(projector_width) / float(projector_height)
		image_ratio = float(image_orig.shape[1]) / float(image_orig.shape[0])
		if projector_ratio > image_ratio:
			# image too portrait, add width
			new_width = round(image_orig.shape[0] * projector_ratio)
			new_height = image_orig.shape[0]

			pad_widths = round(float(new_width - image_orig.shape[1]) / 2)
			pad_heights = 0
		elif projector_ratio < image_ratio:
			# image too landscape, add height
			new_width = image_orig.shape[1]
			new_height = round(image_orig.shape[1] / projector_ratio)

			pad_widths = 0
			pad_heights = round(float(new_height - image_orig.shape[0]) / 2)
		else:
			# same format already, do nothing
			pad_widths = 0
			pad_heights = 0
		# pad image to ratio of second screen, just like the background image is
		image = np.pad(image_orig, ((pad_heights, pad_heights), (pad_widths, pad_widths)))
		image_width, image_height = image.shape[1], image.shape[0]

		# tracking window
		image_diff = cv2.subtract(image, image_bg)  # TODO: black on white or white on black?
		cv2.normalize(image_diff, image_diff, 0, 255, cv2.NORM_MINMAX)
		rot_window, track_window = cv2.CamShift(image_diff, track_window, term_crit)
		track_w_max, track_h_max = max_track_window_size
		track_x, track_y, track_w, track_h = track_window
		(rot_cx, rot_cy), (rot_w, rot_h), angle = rot_window  # angle of the width axis wrt to the x-axis

		if track_w > track_w_max:  # clamp
			track_w = track_w_max
			track_window = track_x, track_y, track_w_max, track_h
		if track_h > track_h_max:
			track_h = track_h_max
			track_window = track_x, track_y, track_w, track_h_max

		if rot_w > rot_h:  # fix angle & shape
			os.system('pause')
			angle = (angle + 90) % 180
			temp = rot_w
			rot_w = rot_h
			rot_h = temp
		if rot_w > track_w_max:
			rot_w = track_w_max
			rot_window = (rot_cx, rot_cy), (track_w_max, rot_h), angle
		if rot_h > track_h_max:
			rot_h = track_h_max
			rot_window = (rot_cx, rot_cy), (rot_w, track_h_max), angle

		# rotation
		roi = image_diff[track_y:track_y+track_h, track_x:track_x+track_w]
		roi_rotated = rotate(roi, angle)  # rotates in the opposite direction
		utils.show_image(roi_rotated, duration=1)
		cv2.imwrite(os.path.join(path, 'processed', f'{int(i_c / int(100. / precision))}.jpg'), roi_rotated)
		print(os.path.join(path, 'processed', f'{i_c}.jpg'))

		# region DISPLAY ===============================

		# convert projector shape to image space
		projector_width_imsp = round(projector_width * projector_image_ratio)
		projector_height_imsp = round(projector_height * projector_image_ratio)

		# zoom
		image_zoomed = cv2.resize(image, (round(image_width * delta_z), round(image_height * delta_z)), interpolation=cv2.INTER_LINEAR)
		image_diff_zoomed = cv2.resize(image_diff, (round(image_width * delta_z), round(image_height * delta_z)), interpolation=cv2.INTER_LINEAR)

		# pad to make canvas
		image_canvas = np.pad(image_zoomed, ((projector_height_imsp, projector_height_imsp), (projector_width_imsp, projector_width_imsp)))
		image_diff_canvas = np.pad(image_diff_zoomed, ((projector_height_imsp, projector_height_imsp), (projector_width_imsp, projector_width_imsp)))

		# move
		image_moved = image_canvas[
			projector_height_imsp - delta_y:projector_height_imsp + projector_height_imsp - delta_y,
			projector_width_imsp - delta_x:projector_width_imsp + projector_width_imsp - delta_x
		]
		image_diff_moved = image_diff_canvas[
			projector_height_imsp - delta_y:projector_height_imsp + projector_height_imsp - delta_y,
			projector_width_imsp - delta_x:projector_width_imsp + projector_width_imsp - delta_x
		]

		# zoom & move rotation & tracking window
		rot_window_zoomed = (rot_cx * delta_z + delta_x, rot_cy * delta_z + delta_y), (rot_w * delta_z, rot_h * delta_z), angle
		rwz_pts = cv2.boxPoints(rot_window_zoomed).astype(np.intp)
		twz_x, twz_y, twz_w, twz_h = track_x * delta_z + delta_x, track_y * delta_z + delta_y, track_w * delta_z, track_h * delta_z
		twz_pts = cv2.boxPoints(((twz_x + twz_w / 2, twz_y + twz_h / 2), (twz_w, twz_h), 0)).astype(np.intp)

		# draw
		color_mono = tuple([255 - int(round(255 * float(sum(color)) / len(color))) for _ in color])
		
		image_track = cv2.polylines(image_moved, [rwz_pts], True, (255, 255, 255), 2)
		image_track = cv2.polylines(image_track, [twz_pts], True, (127, 127, 127), 2)

		image_project = np.ones((image_moved.shape[0], image_moved.shape[1], 3))
		image_diff_track = cv2.polylines(image_diff_moved, [rwz_pts], True, (255, 255, 255), 2)

		# show
		cv2.imshow(trackwindow_name, image_track)
		cv2.setMouseCallback(trackwindow_name, mouse_callback)
		# placeholder
		cv2.imshow(diff_window_name, image_diff_track)
		cv2.setMouseCallback(diff_window_name, mouse_callback)

		# endregion

		end = time.perf_counter()
		cv2.waitKey(frame_delay)

		#print(f'FPS: {1. / (end - start):.0f}')

	# placeholder
	# placeholder
	controller.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
