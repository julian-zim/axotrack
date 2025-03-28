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
from pypylon import pylon
import utils


# constants
#model_name = 'D1-D2-L1_mlp'
#model_name = 'D1-L1_mlp'	# Use recording D2
#model_name = 'D2-L1_mlp'
#model_name = 'D1-D2_mlp'
model_name = 'test2-test4-test5_mlp'

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


def get_orientation_naive(section):
	section_height = section.shape[0]
	section_top = section[:section_height // 2, :]
	section_bottom = section[section_height // 2:, :]
	section_top_max = round(np.sum(section_top))
	section_bottom_max = round(np.sum(section_bottom))
	return section_top_max > section_bottom_max


def get_orientation(section, mlp_model):  # TODO: try contours?
	#before = time.perf_counter()
	w_pad_val = 50 - section.shape[1]
	h_pad_val = 50 - section.shape[0]
	if w_pad_val < 0 or h_pad_val < 0:
		raise ValueError(f'Image larger than 50 x 50!')
	img_padded = np.pad(section, ((w_pad_val // 2, w_pad_val - (w_pad_val // 2)), (h_pad_val // 2, h_pad_val - (h_pad_val // 2))))
	label = mlp_model.predict(img_padded.flatten().astype(np.intp)[None])
	#after = time.perf_counter()
	#print(f'Orientation: {label[0]}, FPS: {1. / (after - before):.0f}')
	return not bool(label[0])


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
	if len(monitors) < 2:
		utils.show_error('Cannot find second monitor.')
		quit()
	first_monitor, second_monitor = monitors

	# get orientation model
	mlp_model = load(f'models/{model_name}.joblib')

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
	projector_width, projector_height = second_monitor.width, second_monitor.height
	projector_image_ratio = float(image_bg_height) / float(projector_height)  # should be same as image_width / projector_width

	# get dot offset file path
	txt_do_path = os.path.join(os.getcwd(), 'data', 'dot_offset.txt')
	if not os.path.exists(txt_do_path):
		txt_do_path = utils.choose_file('No dot offset text file could be found at the default location. Please choose one form your PC.')
	if not txt_do_path:
		utils.show_error('Problem choosing file.')
		quit()
	with open(txt_do_path) as file:
		delta = file.readline()[:-1].split(',')
		try:
			delta_x, delta_y = int(delta[0]), int(delta[1])
			delta_z = float(file.readline()[:-1])
		except (ValueError, IndexError):
			utils.show_error(f'The dot offset txt file at \"{txt_do_path}\" is corrupted.')

	# setup windows
	proj_window_name = 'Projector'
	cv2.namedWindow(proj_window_name, cv2.WND_PROP_FULLSCREEN)
	cv2.moveWindow(proj_window_name, second_monitor.x, second_monitor.y)
	cv2.setWindowProperty(proj_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	trackwindow_name = 'Tracking View'
	trackwindow_width, trackwindow_height = round(projector_width / trackwindow_screen_ratio), round(projector_height / trackwindow_screen_ratio)
	cv2.namedWindow(trackwindow_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(trackwindow_name, trackwindow_width, trackwindow_height)
	cv2.moveWindow(trackwindow_name, first_monitor.x + (first_monitor.height - trackwindow_height) // 2, first_monitor.y + (first_monitor.height - trackwindow_height) // 2)

	diff_window_name = 'Difference View'
	diff_window_width, diff_window_height = int(round(projector_width / diff_window_screen_ratio)), round(projector_height / diff_window_screen_ratio)
	cv2.namedWindow(diff_window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(diff_window_name, diff_window_width, diff_window_height)
	cv2.moveWindow(diff_window_name, first_monitor.width * 3 // 4 - diff_window_width // 2, first_monitor.y + first_monitor.height // 2)

	# setup tracking
	global track_window
	track_window = (image_bg_width // 2 - max_track_window_size[0] // 2, image_bg_height // 2 - max_track_window_size[1] // 2, max_track_window_size[0], max_track_window_size[1])
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

	# setup camera
	camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
	camera.Open()
	if not camera.IsOpen():
		print('Cannot find camera.')
		return
	utils.setupCamera(camera)

	# control panel
	controller.show()

	#endregion

	# main loop
	frame_delay = round(1000. / max_fps)
	noise = (0., 0.)
	counter = 0
	camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
	while True:
		start = time.perf_counter()

		# quit
		if cv2.getWindowProperty(trackwindow_name, cv2.WND_PROP_VISIBLE) + cv2.getWindowProperty(diff_window_name, cv2.WND_PROP_VISIBLE) < 2 or not controller.isVisible():
			break

		# read camera
		grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		if not grabResult.GrabSucceeded(): break
		image_orig = cv2.flip(grabResult.Array, 1)
		if len(image_orig.shape) != 2:
			image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)

		# image preprocessing (pad up to second screen size)
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
		image = np.pad(image_orig, ((pad_heights, pad_heights), (pad_widths, pad_widths)))
		image_width, image_height = image.shape[1], image.shape[0]

		# tracking
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

		# fix angle & shape
		#print(f'ANGLE: {angle:.2f}, WIDTH: {rot_w:.2f}, HEIGHT: {rot_h:.2f}')
		if rot_w > rot_h:  
			#os.system('pause')
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
		#orientation = get_orientation_naive(roi_rotated)
		orientation = get_orientation(roi_rotated, mlp_model)
		if orientation:
			angle = (angle + 180) % 360
		#print(f'width: {width:.2f}, height: {height:.2f}, wa: {angle:.2f}, upper_hemisphere: {upper_hemisphere}')

		# dot position
		center = np.array((rot_cx, rot_cy))
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
		dot += noise

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

		# zoom & move dot
		dot = (dot[0] * delta_z + delta_x, dot[1] * delta_z + delta_y)
		center = (center[0] * delta_z + delta_x, center[1] * delta_z + delta_y)

		# zoom & move rotation & tracking window
		rot_window_zoomed = (rot_cx * delta_z + delta_x, rot_cy * delta_z + delta_y), (rot_w * delta_z, rot_h * delta_z), angle
		rwz_pts = cv2.boxPoints(rot_window_zoomed).astype(np.intp)
		twz_x, twz_y, twz_w, twz_h = track_x * delta_z + delta_x, track_y * delta_z + delta_y, track_w * delta_z, track_h * delta_z
		twz_pts = cv2.boxPoints(((twz_x + twz_w / 2, twz_y + twz_h / 2), (twz_w, twz_h), 0)).astype(np.intp)

		# draw
		color_mono = tuple([255 - int(round(255 * float(sum(color)) / len(color))) for _ in color])
		
		image_track = cv2.polylines(image_moved, [rwz_pts], True, (255, 255, 255), 2)
		image_track = cv2.polylines(image_track, [twz_pts], True, (127, 127, 127), 2)
		image_track_line = cv2.line(image_track, [round(c) for c in center], [round(c) for c in dot], color_mono, 1)
		image_track_dot = cv2.circle(image_track_line, [round(c) for c in dot], round(size), color_mono, -1)

		image_project = np.ones((image_moved.shape[0], image_moved.shape[1], 3))
		image_project_dot = cv2.circle(image_project, [round(c) for c in dot], round(size), color, -1) if active else image_project
		image_diff_track = cv2.polylines(image_diff_moved, [rwz_pts], True, (255, 255, 255), 2)

		# show
		cv2.imshow(trackwindow_name, image_track_dot)
		cv2.setMouseCallback(trackwindow_name, mouse_callback)
		cv2.imshow(proj_window_name, image_project_dot)
		cv2.imshow(diff_window_name, image_diff_track)
		cv2.setMouseCallback(diff_window_name, mouse_callback)

		# endregion

		end = time.perf_counter()
		cv2.waitKey(frame_delay)

		print(f'FPS: {1. / (end - start):.0f}')

	camera.StopGrabbing()
	camera.Close()
	controller.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
