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
from joblib import load
# placeholder
import utils

from tracking_alternative import preprocessing, get_roi, roi_backrotation, get_orientation, get_orientation_naive


camera_fps = 30
max_fps = 30
trackwindow_screen_ratio = 1.5
diff_window_screen_ratio = 2.5

default_names = [
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
	'0',  # 30
	'15', # 15
	'0,0,0',
	'5',  # 2
	'0',  # 50
	'0'  # 10
]

# global variables
track_window = (0, 0, default_values[0], default_values[1])


class ParameterControl(QWidget):
	parameter_signal = pyqtSignal(*[str for _ in range(len(default_names))])

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Control Panel")
		self.resize(250, 198)  # tweak

		layout = QVBoxLayout()
		self.labels = [QLabel(default_title) for default_title in default_names]
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


def main():
	# region VARIABLES
	track_window_size = (int(default_values[0]), int(default_values[1]))
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
		nonlocal track_window_size, active, periphery, distance_x, distance_y, color, size, jitter_speed, jitter_magnitude
		track_window_size = (int(tracker_width_new), int(tracker_height_new))
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
	first_monitor = monitors[0]

	# get background image
	image_bg_path = os.path.join('.', 'data', 'background.jpg')
	if not os.path.exists(image_bg_path):
		image_bg_path = utils.choose_file('No background image file could be found at the default location. Please choose one form your PC.')
	if not image_bg_path:
		utils.show_error('Problem choosing file.')
		quit()
	image_bg = cv2.imread(image_bg_path)
	if len(image_bg.shape) != 2:
		image_bg = cv2.cvtColor(image_bg, cv2.COLOR_BGR2GRAY)
	image_bg_width, image_bg_height = image_bg.shape[1], image_bg.shape[0]

	# setup windows
	trackwindow_name = 'Tracking View'
	trackwindow_width, trackwindow_height = round(image_bg_width / trackwindow_screen_ratio), round(image_bg_height / trackwindow_screen_ratio)
	cv2.namedWindow(trackwindow_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(trackwindow_name, trackwindow_width * 2, trackwindow_height * 2)
	cv2.moveWindow(trackwindow_name, first_monitor.x + (first_monitor.height - trackwindow_height) // 2, first_monitor.y + (first_monitor.height - trackwindow_height) // 2)

	diff_window_name = 'Difference View'
	diff_window_width, diff_window_height = int(round(image_bg_width / diff_window_screen_ratio)), round(image_bg_height / diff_window_screen_ratio)
	cv2.namedWindow(diff_window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(diff_window_name, diff_window_width * 2, diff_window_height * 2)
	cv2.moveWindow(diff_window_name, first_monitor.width * 3 // 4 - diff_window_width // 2, first_monitor.y + first_monitor.height // 2)

	# setup tracking
	global track_window
	#track_window_initial = (image_bg_width // 2 - track_window_size[0] // 2, image_bg_height // 2 - track_window_size[1] // 2, track_window_size[0], track_window_size[1])
	track_window = (263, 196, 45, 47)
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

	# setup camera
	path = utils.choose_directory()
	img_names = sorted(os.listdir(path), key=lambda name: int(name.split('_')[-1].split('.')[0]))
	num_imgs = len(img_names)
	i_c = 0

	# control panel
	controller.show()

	#endregion

	# main loop
	frame_delay = round(1000. / max_fps)
	noise = (0., 0.)
	counter = 0
	# placeholder
	while i_c < num_imgs:
		start = time.perf_counter()

		# quit
		if cv2.getWindowProperty(trackwindow_name, cv2.WND_PROP_VISIBLE) + cv2.getWindowProperty(diff_window_name, cv2.WND_PROP_VISIBLE) < 2 or not controller.isVisible():
			break

		# read camera
		frame = cv2.imread(os.path.join(path, img_names[i_c]))
		i_c += round(max_fps / camera_fps)
		image_raw = frame
		if len(image_raw.shape) != 2:
			image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)

		# tracking
		image = preprocessing(image_raw, image_bg)
		#cv2.imwrite(f'C:/Users/zimmermann.admin/Desktop/prepr/{i_c}.jpg', image)
		rot_window, track_window = get_roi(image, track_window, track_window_size, term_crit)
		(tw_x, tw_y, tw_w, tw_h) = track_window
		(rot_cx, rot_cy), (rot_w, rot_h), angle = rot_window

		# get rotation
		roi = image[tw_y:tw_y+tw_h, tw_x:tw_x+tw_w]
		roi_rotated = roi_backrotation(roi, angle)  # rotates in the opposite direction
		cv2.imwrite(f'C:/Users/zimmermann.admin/Desktop/roi-backrot/{i_c}.jpg', roi_rotated)
		orientation = get_orientation_naive(roi_rotated)
		if orientation:
			angle = (angle + 180) % 360

		# dot position TODO
		'''center = np.array((rot_cx, rot_cy))
		offset_x = np.array(utils.pol2cart(distance_x, np.deg2rad((angle))))
		offset_y = np.array(utils.pol2cart(max([track_w, track_h]) / 2 + distance_y, np.deg2rad(angle + 90) % 360))
		dot = center + offset_x * (1 if periphery else -1) + offset_y
		if jitter_speed > 0:
			counter += 1
			jitter_speed_log = round(math.log10(jitter_speed) * 50)  # why 50??? but ey, works      # also, maybe just do 10 / speed?
			if counter >= (100 - jitter_speed_log):
				noise = np.array((random.randrange(-jitter_magnitude, jitter_magnitude + 1), random.randrange(-jitter_magnitude, jitter_magnitude + 1)))
				counter = 0
		else:
			noise = (0., 0.)
		dot += noise'''
	
		# draw
		twz_pts = cv2.boxPoints(((tw_x + tw_w / 2, tw_y + tw_h / 2), (tw_w, tw_h), 0)).astype(np.intp)
		image_raw_track = cv2.polylines(image_raw, [twz_pts], True, (127, 127, 127), 2)
		image_track = cv2.polylines(image, [twz_pts], True, (127, 127, 127), 2)

		# show
		cv2.imshow(trackwindow_name, image_raw_track)
		cv2.setMouseCallback(trackwindow_name, mouse_callback)
		# placeholder
		cv2.imshow(diff_window_name, image_track)
		cv2.setMouseCallback(diff_window_name, mouse_callback)

		# endregion
		
		end = time.perf_counter()
		cv2.waitKey(frame_delay)

		print(f'FPS: {1. / (end - start):.0f}')

	# placeholder
	# placeholder
	controller.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
