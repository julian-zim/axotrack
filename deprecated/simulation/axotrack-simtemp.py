import sys
import os
import time
import math
import numpy as np
import cv2
from screeninfo import get_monitors
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QSlider, QCheckBox
from PyQt5.QtGui import QColor, QDoubleValidator
#from pypylon import pylon
import imageio
import utils


image_size = (1000, 1000)  # (1544, 1544)
window_types = [
	'raw',
	'diff (raw)',
	'diff (norm)',
	'binary (body)',
	'binary (eyes)'
]

window_type = 2


def mouse_callback(event, x, y, flags, param):
	global window_type

	if event == cv2.EVENT_LBUTTONUP:
		window_type = (window_type + 1) % len(window_types)

	if event == cv2.EVENT_RBUTTONUP:
		window_type = (window_type - 1) % len(window_types)


class ParameterControl(QWidget):
	parameter_signal = pyqtSignal(int, int, bool, float, float, float, tuple, int, float, float)
	

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Control Panel")
		
		validator = QDoubleValidator()
		validator.setNotation(QDoubleValidator.StandardNotation)
		layout_main = QVBoxLayout()


		# thresholds
		self.thresholds_labels = (QLabel('Body Threshold:'), QLabel('Eye Threshold:'))
		for label in self.thresholds_labels: label.setFixedSize(100, 20) 

		self.thresholds = (QSlider(Qt.Horizontal), QSlider(Qt.Horizontal))
		for slider in self.thresholds: slider.setMinimum(0)
		for slider in self.thresholds: slider.setMaximum(255)
		self.thresholds[0].setValue(20)
		self.thresholds[1].setValue(200)
		for slider in self.thresholds: slider.valueChanged.connect(self.emit_values)

		for i in range(2):
			layout = QHBoxLayout()
			layout.addWidget(self.thresholds_labels[i])
			layout.addWidget(self.thresholds[i])
			layout_main.addLayout(layout)

		# dot active
		self.active_label = QLabel('Dot Active:')
		self.active_label.setFixedSize(100, 20) 
	
		self.active = QCheckBox()
		self.active.setChecked(False)
		self.active.stateChanged.connect(self.emit_values)

		layout = QHBoxLayout()
		layout.addWidget(self.active_label)
		layout.addWidget(self.active)
		layout_main.addLayout(layout)

		# dot distances
		self.distances_labels = (QLabel('Medial Distance:'), QLabel('Lateral Distance:'))
		for label in self.distances_labels: label.setFixedSize(100, 20)

		self.distances = (QLineEdit(), QLineEdit())
		for edit in self.distances: edit.setValidator(validator)
		self.distances[0].setText('60')
		self.distances[1].setText('0')
		for edit in self.distances: edit.editingFinished.connect(self.emit_values)

		for i in range(2):
			layout = QHBoxLayout()
			layout.addWidget(self.distances_labels[i])
			layout.addWidget(self.distances[i])
			layout_main.addLayout(layout)

		# dot size
		self.radius_label = QLabel('Size:')
		self.radius_label.setFixedSize(100, 20) 

		self.radius = QLineEdit()
		self.radius.setValidator(validator)
		self.radius.setText('5')
		self.radius.editingFinished.connect(self.emit_values)

		layout = QHBoxLayout()
		layout.addWidget(self.radius_label)
		layout.addWidget(self.radius)
		layout_main.addLayout(layout)

		# dot color
		self.hue_label = QLabel('Color:')
		self.hue_label.setFixedSize(100, 20)

		self.hue = QSlider(Qt.Horizontal)
		self.hue.setMinimum(0)
		self.hue.setMaximum(255)
		self.hue.setValue(127)
		self.hue.valueChanged.connect(self.emit_values)

		layout = QHBoxLayout()
		layout.addWidget(self.hue_label)
		layout.addWidget(self.hue)
		layout_main.addLayout(layout)

		# dot brightness
		self.brightness_label = QLabel('Brightness:')
		self.brightness_label.setFixedSize(100, 20)

		self.brightness = QSlider(Qt.Horizontal)
		self.brightness.setMinimum(0)
		self.brightness.setMaximum(511)
		self.brightness.setValue(255)
		self.brightness.valueChanged.connect(self.emit_values)

		layout = QHBoxLayout()
		layout.addWidget(self.brightness_label)
		layout.addWidget(self.brightness)
		layout_main.addLayout(layout)

		# dot jitter speed
		self.jitter_speed_label = QLabel('Jitter Speed:')
		self.jitter_speed_label.setFixedSize(100, 20) 

		self.jitter_speed = QSlider(Qt.Horizontal)
		self.jitter_speed.setMinimum(0)
		self.jitter_speed.setMaximum(100)
		self.jitter_speed.setValue(90)
		self.jitter_speed.valueChanged.connect(self.emit_values)

		layout = QHBoxLayout()
		layout.addWidget(self.jitter_speed_label)
		layout.addWidget(self.jitter_speed)
		layout_main.addLayout(layout)

		# dot jitter strength
		self.jitter_magnitudes_labels = (QLabel('Medial Jitter:'), QLabel('Lateral Jitter:'))
		for label in self.jitter_magnitudes_labels: label.setFixedSize(100, 20) 

		self.jitter_magnitudes = (QLineEdit(), QLineEdit())
		for edit in self.jitter_magnitudes: edit.setValidator(validator)
		self.jitter_magnitudes[0].setText('5')
		self.jitter_magnitudes[1].setText('10')
		for edit in self.jitter_magnitudes: edit.editingFinished.connect(self.emit_values)


		for i in range(2):
			layout = QHBoxLayout()
			layout.addWidget(self.jitter_magnitudes_labels[i])
			layout.addWidget(self.jitter_magnitudes[i])
			layout_main.addLayout(layout)

		self.setLayout(layout_main)
		self.setMinimumWidth(240)
		self.adjustSize()

		_, _, screen_width, screen_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(screen_width * 3 // 4 - width // 2, screen_height // 2 - height // 2)


	def emit_values(self):
		radius_float = utils.string_to_float(self.radius.text())
		if radius_float < 1.:
			self.radius.setText('1')
			radius_float = 1.

		color = QColor()
		color.setHsv(self.hue.value(), 255 - max(0, self.brightness.value() - 256), min(self.brightness.value(), 255))

		jitter_magnitudes_float = [-1., -1.]
		for i in range(2):
			jitter_magnitudes_float[i] = utils.string_to_float(self.jitter_magnitudes[i].text())
			if jitter_magnitudes_float[i] < 0.:
				self.jitter_magnitudes[i].setText('0')
				jitter_magnitudes_float[i] = 0.

		self.parameter_signal.emit(
			self.thresholds[0].value(),
			self.thresholds[1].value(),
			self.active.isChecked(),
			utils.string_to_float(self.distances[0].text()),
			utils.string_to_float(self.distances[1].text()),
			radius_float,
			color.getRgb()[:3],
			self.jitter_speed.value(),
			jitter_magnitudes_float[0],
			jitter_magnitudes_float[1]
		)


	def closeEvent(self, event):
		QApplication.exit(0)


def main():
	
	# region PyQT Stuff
	threshold_body = -1
	threshold_eyes = -1
	active = None
	distance_y = -1.
	distance_x = -1.
	radius = -1
	color = (-1, -1, -1)
	jitter_speed = -1
	jitter_magnitude_y = -1
	jitter_magnitude_x = -1

	def update_values(
			threshold_body_new,
			threshold_eyes_new,
			active_new,
			distance_y_new,
			distance_x_new,
			radius_new,
			color_new,
			jitter_speed_new,
			jitter_magnitude_y_new,
			jitter_magnitude_x_new,
		):
		nonlocal threshold_body, threshold_eyes, active, distance_y, distance_x, radius, color, jitter_speed, jitter_magnitude_y, jitter_magnitude_x
		threshold_body = threshold_body_new
		threshold_eyes = threshold_eyes_new
		active = active_new
		distance_x = distance_x_new
		distance_y = distance_y_new
		color = color_new
		radius = round(radius_new)
		jitter_speed = jitter_speed_new
		jitter_magnitude_y = round(jitter_magnitude_y_new)
		jitter_magnitude_x = round(jitter_magnitude_x_new)

	app = QApplication(sys.argv)
	controller = ParameterControl()
	controller.parameter_signal.connect(update_values)
	controller.emit_values()
	controller.show()
	# endregion

	cv2.setUseOptimized(True)

	# background
	filepath = os.path.join(os.getcwd(), 'data', f'background.jpg')
	if not os.path.exists(filepath):
		filepath = utils.choose_file('No background image was found. Please specify one manually.')
	image_bg = cv2.imread(filepath)
	if image_bg is None:
		utils.show_error('Cannot read background image.')
		return
	image_bg = cv2.cvtColor(image_bg, cv2.COLOR_BGR2GRAY)
	image_bg = cv2.resize(image_bg, image_size)

	# window setup
	monitors = get_monitors()
	trackwindow_name = 'Tracking Window (click to change view)'
	cv2.namedWindow(trackwindow_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(trackwindow_name, monitors[0].height * 8 // 10, monitors[0].height * 8 // 10)
	cv2.moveWindow(trackwindow_name, monitors[0].x + monitors[0].height * 1 // 10, monitors[0].y + monitors[0].height * 1 // 10)
	cv2.setMouseCallback(trackwindow_name, mouse_callback)
	if len(monitors) > 1:
		projwindow_name = 'Projection Window'
		cv2.namedWindow(projwindow_name, cv2.WINDOW_NORMAL)
		cv2.moveWindow(projwindow_name, monitors[1].x, monitors[1].y)
		cv2.setWindowProperty(projwindow_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		image_proj = np.full((image_size[1], image_size[1], 3), 255, dtype=np.uint8)
		pad_width = round((image_size[1] * monitors[1].width / monitors[1].height - image_size[0]) / 2)
		image_proj = np.pad(image_proj, ((0, 0), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
	else:
		utils.show_info('No second screen detected!')

	# in- & output setup
	#camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
	#camera.Open()
	#if not camera.IsOpen():
		#utils.show_error('Cannot find camera.')
		#return
	#utils.setupCamera(camera)
	outfile_path = utils.save_file('Choose a video output file name & location.', os.getcwd(), [('MP4', '.mp4')], '.mp4', 'output.mp4')
	video_writer = imageio.get_writer(os.path.join(os.getcwd(), outfile_path), fps=30)
	dirpath = utils.choose_directory('Select input video directory.')
	img_names = sorted(os.listdir(dirpath), key=lambda name: int(name.split('_')[-1].split('.')[0]))
	num_imgs = len(img_names)
	i_c = 0  # poi: 1000, 1730

	# main loop
	dot_position_prev = np.full(2, -1.)
	noise = np.zeros(2)
	jitter_counter = 0

	#camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
	while i_c < num_imgs:  #camera.IsGrabbing()
		before = time.perf_counter()

		# quit
		if not (cv2.getWindowProperty(trackwindow_name, cv2.WND_PROP_VISIBLE) and controller.isVisible()): break


		# read & preprocess
		#grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		#if not grabResult.GrabSucceeded(): break
		#frame = cv2.flip(grabResult.Array, -1)
		frame = cv2.flip(cv2.imread(os.path.join(dirpath, img_names[i_c])), -1)
		#grabResult.Release()
		if len(frame.shape) != 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		image_raw = cv2.resize(frame, image_size)
		image_diff_raw = cv2.absdiff(image_raw, image_bg)
		image_diff = cv2.normalize(image_diff_raw, None, 0, 255, cv2.NORM_MINMAX) # type: ignore


		# body
		_, image_bin_body = cv2.threshold(image_diff, threshold_body, 255, cv2.THRESH_BINARY)
		image_bin_body = cv2.morphologyEx(image_bin_body, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)   # erode & dilate

		body_centroid = np.full(2, -1.)
		moments = cv2.moments(image_bin_body)
		if moments['m00'] > 0:
			body_centroid = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])


		# eyes
		_, image_bin_eyes = cv2.threshold(image_diff, threshold_eyes, 255, cv2.THRESH_BINARY)

		contours, _ = cv2.findContours(image_bin_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) > 2:
			print('Warning! More than two eyes detected?')

		eyes_centroids = np.full((2, 2), -1.)
		if len(contours) >= 2:
			contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # two largest contours (hopefully the eyes)

			moments = (cv2.moments(contours_sorted[0]), cv2.moments(contours_sorted[1]))
			if moments[0]['m00'] > 0:
				eyes_centroids[0] = (np.array([moments[0]['m10'] / moments[0]['m00'], moments[0]['m01'] / moments[0]['m00']]))
			if moments[1]['m00'] > 0:
				eyes_centroids[1] = (np.array([moments[1]['m10'] / moments[1]['m00'], moments[1]['m01'] / moments[1]['m00']]))

		eyes_found = not any((eyes_centroids < 0.).flatten())


		# dot
		if eyes_found:  # in case of eyes detected, use eye orientation for dot position and jitter orientation
			eyes_centroids = eyes_centroids[np.argsort(eyes_centroids[:, 0])]
			eyes_center = np.mean(eyes_centroids, axis=0)

			lateral_vector = eyes_centroids[1] - eyes_centroids[0]
			lateral_vector /= np.linalg.norm(lateral_vector)
			medial_vector = np.array([lateral_vector[1], -lateral_vector[0]])
			
			orientation = np.sign(np.dot(eyes_center - body_centroid, medial_vector))
			
			dot_position = eyes_center + (medial_vector * distance_y + lateral_vector * distance_x) * orientation
			dot_position_prev = dot_position
		else:  # in case of eyes not detected, keep last dot position and use relative orientation towards it as jitter orientation
			dot_position = dot_position_prev.copy()
			medial_vector = dot_position - body_centroid
			medial_vector /= np.linalg.norm(medial_vector)
			lateral_vector = np.array([medial_vector[1], -medial_vector[0]])

		# jitter
		if jitter_speed > 0:
			jitter_counter += 1
			jitter_speed_log = round(math.log10(jitter_speed) * 50)  # why 50??? but ey, works      # also, maybe just do 10 / speed?
			if jitter_counter >= (100 - jitter_speed_log):
				noise = medial_vector * np.random.randint(-jitter_magnitude_y, jitter_magnitude_y + 1) + lateral_vector * np.random.randint(-jitter_magnitude_x, jitter_magnitude_x + 1)
				jitter_counter = 0
			dot_position += noise


		# draw
		match window_type:
			case 0:
				image_vis = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2BGR)
			case 1:
				image_vis = cv2.cvtColor(image_diff_raw, cv2.COLOR_GRAY2BGR)
			case 2:
				image_vis = cv2.cvtColor(image_diff, cv2.COLOR_GRAY2BGR)
			case 3:
				image_vis = cv2.cvtColor(image_bin_body, cv2.COLOR_GRAY2BGR)
			case 4:
				image_vis = cv2.cvtColor(image_bin_eyes, cv2.COLOR_GRAY2BGR)
			case _:
				image_vis = cv2.cvtColor(image_diff, cv2.COLOR_GRAY2BGR)

		# main screen
		cv2.circle(image_vis, tuple(np.rint(body_centroid).astype(np.intp)), color=(200, 200, 200), radius=5)  # body centroid
		if eyes_found:
			cv2.circle(image_vis, tuple(np.rint(eyes_centroids[0]).astype(np.intp)), color=(0, 127, 255), radius=2, thickness=-1)  # eye 1 centroid
			cv2.circle(image_vis, tuple(np.rint(eyes_centroids[1]).astype(np.intp)), color=(0, 127, 255), radius=2, thickness=-1)  # eye 2 centroid
			cv2.circle(image_vis, tuple(np.rint(eyes_center).astype(np.intp)), color=(0, 127, 255), radius=2, thickness=-1)  # eyes center
			cv2.line(image_vis, tuple(np.rint(eyes_centroids[0]).astype(np.intp)), tuple(np.rint(eyes_centroids[1]).astype(np.intp)), color=(0, 0, 255), thickness=1)  # eyes vector
			cv2.line(image_vis, tuple(np.rint(dot_position).astype(np.intp)), tuple(np.rint(eyes_center).astype(np.intp)), color=(0, 0, 255), thickness=1)  # eyes vector orth 
		cv2.circle(image_vis, tuple(np.rint(dot_position).astype(np.intp)), color=color, radius=radius, thickness=-1)  # dot

		cv2.putText(image_vis, f'{str(window_type)}: {window_types[window_type]}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
		cv2.imshow(trackwindow_name, image_vis)

		# projector
		if len(monitors) > 1:
			if active:
				cv2.circle(image_proj, tuple(np.rint((dot_position[0] + pad_width, dot_position[1])).astype(np.intp)), color=color, radius=radius, thickness=-1)  # draw dot
			cv2.imshow(projwindow_name, image_proj)
			if active:
				cv2.circle(image_proj, tuple(np.rint((dot_position[0] + pad_width, dot_position[1])).astype(np.intp)), color=(255, 255, 255), radius=radius, thickness=-1)  # remove dot again for next draw

		# video
		image_dot = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2BGR)
		if active:
			cv2.circle(image_dot, tuple(np.rint((dot_position[0], dot_position[1])).astype(np.intp)), color=color, radius=radius, thickness=-1)  # draw dot
		video_writer.append_data(image_dot)  # TODO: manage to write in fixed time interval, for a 30fps video output


		# print & end
		after = time.perf_counter()
		cv2.waitKey(1)
		#print(f'{1 / (after - before):.2f}fps - {(after - before)*1000:.2f}ms ' + ('' if eyes_found else '- no eyes found'))
		print(f'{i_c}/{num_imgs} - {1 / (after - before):.2f}fps - {(after - before)*1000:.2f}ms ' + ('' if eyes_found else '- no eyes found'))
		i_c += 1

	#camera.StopGrabbing()
	#camera.Close()
	video_writer.close()
	cv2.destroyAllWindows()
	app.quit()


if __name__ == '__main__':
	main()
