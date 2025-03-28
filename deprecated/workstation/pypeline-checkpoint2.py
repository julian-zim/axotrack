# 200 fps wtf??


import time
import sys
import os
import random
import math
import timeit
import numpy as np
from matplotlib import pyplot as plt
import cv2
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QDesktopWidget, QSlider
import utils


default_names = [
	'Body Threshold',
	'Eyes Threshold',
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
	'15',
	'200',
	'50',
	'50',
	'0',
	'-1',
	'0',  # 30
	'60', # 15
	'0,255,0',
	'5',  # 2
	'0',  # 50
	'0'  # 10
]

class ParameterControl(QWidget):
	parameter_signal = pyqtSignal(*[int, int] + [str for _ in range(len(default_names)-2)])  # EXCEPTION

	def __init__(self):
		super().__init__()
		self.setWindowTitle("Control Panel")
		self.resize(250, 198)  # tweak

		layout = QVBoxLayout()
		self.labels = [QLabel(default_name) for default_name in default_names]
		for label in self.labels:
			label.setFixedSize(100, 20)  # tweak
		self.edits = [QLineEdit(default_value) for default_value in default_values]
		# == EXCEPTIONS: make first & second lineedit a slider ==
		self.edits[0] = QSlider(Qt.Horizontal)
		self.edits[0].setMinimum(0)
		self.edits[0].setMaximum(255)
		self.edits[0].setValue(int(default_values[0]))
		self.edits[1] = QSlider(Qt.Horizontal)
		self.edits[1].setMinimum(0)
		self.edits[1].setMaximum(255)
		self.edits[1].setValue(int(default_values[1]))
		# =======================================================
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
			# === EXCEPTION: make first & second line edit a slider ===
			if i in [0, 1]:
				self.edits[i].valueChanged.connect(self.emit_values)
			else:
			# =========================================================
				self.edits[i].editingFinished.connect(self.emit_values)

	def emit_values(self):
		# for body & eyes threshold
		for i in [0, 1]:
			try:
				ival = int(self.edits[i].value())
			except (ValueError, AssertionError):
				self.edits[i].setValue(int(default_values[i]))
		# for tracker size
		for i in [2, 3]:
			try:
				ival = int(self.edits[i].text())
				assert(ival > 0)
			except (ValueError, AssertionError):
				self.edits[i].setText(default_values[i])
		# for active
		i = 4
		if self.edits[i].text().upper() not in ['0', '1']:
			self.edits[i].setText(default_values[i])
		# for periphery
		i = 5
		if self.edits[i].text().upper() not in ['-1', '1']:
			self.edits[i].setText(default_values[i])
		# for distance
		for i in [6, 7]:
			try:
				float(self.edits[i].text())
			except ValueError:
				self.edits[i].setText(default_values[i])
		# for color
		i = 8
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
		for i in [9, 11]:
			try:
				fval = float(self.edits[i].text())
				assert(fval >= 0.)
			except (ValueError, AssertionError):
				self.edits[i].setText(default_values[i])
		# for speed
		i = 10
		try:
			fval = float(self.edits[i].text())
			assert(0. <= fval <= 100.)
		except (ValueError, AssertionError):
			self.edits[i].setText(default_values[i])
	
		self.parameter_signal.emit(*[edit.value() if i in [0, 1] else edit.text() for i, edit in enumerate(self.edits)])  # EXCEPTION

	def closeEvent(self, event):
		QApplication.exit(0)

image_size = (1544, 1544)  # (1544, 1544)
tw_size = (image_size[0] // 4, image_size[1] // 4)  # (400, 400)
track_window = (image_size[1] // 2 - tw_size[1] // 2, 0, *tw_size)  # (image_size[1] // 2 - tw_size[1] // 2, image_size[0] // 2 - tw_size[0] // 2, *tw_size)

def mouse_callback(event, x, y, flags, param):
	global track_window

	if event == cv2.EVENT_LBUTTONDOWN:
		track_window = (x, y, track_window[2], track_window[3])


# Unused
def pca(image, threshold=int(default_values[0])):
	# Usage:
	#mean, axis = pca(image_bin_body, threshold=threshold_body)
	#angle = np.degrees(utils.cart2pol(*axis)[1])

	coords = np.argwhere(image >= threshold).astype(np.float32)  # converts format from (i, j) to x = j, y = i)
	if len(coords) < 2:
		return np.array([0., 0.]), np.array([1., 0.])
	means, eigenvectors = cv2.PCACompute(coords, mean=np.empty((0)), maxComponents=1)
	mean = np.array([means[0][1], means[0][0]])
	eigenvector = eigenvectors[0][[1, 0]]  # convert format back from (x, y) to (i = y, j = x)
	return mean, eigenvector


def kmeans(image, threshold=int(default_values[1]), k=2):
	# convert pixels to coordinates
	coords = np.argwhere(image >= threshold).astype(np.uint32)  # converts format from (i, j) to x = j, y = i)

	# ===== DOWNSCALE IMAGE & CONTOURS ======
	coords_offset = np.min(coords, axis=0)  # TODO: also use this to speed up kmeans clustering
	coords_centered = coords - coords_offset
	image_centered = np.zeros(np.max(coords_centered, axis=0) + 1, dtype=np.uint8)
	image_centered[coords_centered[:, 0], coords_centered[:, 1]] = 255
	# check whether two distinct eyes are visible
	contours, _ = cv2.findContours(image_centered, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) < 2:
		return -1, np.empty((0, 0)), np.ones((2, 2)) * -1
	# ========================================

	# get centers of the two largest contours, using moments
	centers = list()
	for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
		moments = cv2.moments(contour)
		if moments['m00'] != 0:
			centers.append(np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']]))
		else:
			centers.append(np.array([-1., -1.]))
		pass
	return 1, _, np.array(centers) + coords_offset[[1, 0]]  # convert format back from (x, y) to (i = y, j = x)
	
	# Old approach: User k-Means clsutering (might be more robust)
	'''
	# clustering
	labels = np.ones(coords.shape[0], dtype=np.int32) * -1
	compactness, _, centers = cv2.kmeans(
		coords.astype(np.float32),
		k,
		labels,
		(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
		1,
		flags=cv2.KMEANS_PP_CENTERS
	)
	
	centers[:, [0, 1]] = centers[:, [1, 0]]  # convert format back from (x, y) to (i = y, j = x)
	return compactness, labels, centers
	'''


def show(window, image, delay=0):
	cv2.imshow(window, image)
	cv2.waitKey(delay)



def main():
	# region PyQT Stuff
	threshold_body = int(default_values[0])
	threshold_eyes = int(default_values[1])
	max_track_window_size = (int(default_values[2]), int(default_values[3]))
	active = bool(int(default_values[4]))
	periphery = (int(default_values[5]) + 1) // 2
	distance_x = float(default_values[6])
	distance_y = float(default_values[7])
	dot_color = (int(default_values[8].split(',')[2]), int(default_values[8].split(',')[1]), int(default_values[8].split(',')[0]))
	dot_size = float(default_values[9])
	jitter_speed = float(default_values[10])
	jitter_magnitude = round(float(default_values[11]))

	def update_values(
			threshold_body_new,
			threshold_eyes_new,
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
		nonlocal threshold_body, threshold_eyes, max_track_window_size, active, periphery, distance_x, distance_y, dot_color, dot_size, jitter_speed, jitter_magnitude
		threshold_body = int(threshold_body_new)
		threshold_eyes = int(threshold_eyes_new)
		max_track_window_size = (int(tracker_width_new), int(tracker_height_new))
		active = bool(int(active_new))
		periphery = (int(periphery_new) + 1) // 2
		distance_x = float(distance_x_new)
		distance_y = float(distance_y_new)
		dot_color = (int(color_new.split(',')[2]), int(color_new.split(',')[1]), int(color_new.split(',')[0]))
		dot_size = float(size_new)
		jitter_speed = float(jitter_speed_new)
		jitter_magnitude = round(float(jitter_magnitude_new))

	app = QApplication(sys.argv)
	controller = ParameterControl()
	controller.parameter_signal.connect(update_values)
	controller.show()
	controller.move(1500, 300)
	# endregion


	#path = 'C:\\Users\\zimmermann.admin\\Desktop\\in\\eyes\\white1\\'
	path = utils.choose_directory()
	img_bg = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(path, '../bg.jpg')), cv2.COLOR_BGR2GRAY), image_size)
	img_names = sorted(os.listdir(path), key=lambda name: int(name.split('_')[-1].split('.')[0]))
	num_imgs = len(img_names)
	global track_window
	trackwindow_name = 'trackwindow'
	cv2.namedWindow(trackwindow_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(trackwindow_name, 700, 700)
	cv2.moveWindow(trackwindow_name, 25, 25)
	cv2.setMouseCallback(trackwindow_name, mouse_callback)
	eyewindow_name = 'eyewindow'
	cv2.namedWindow(eyewindow_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(eyewindow_name, 700, 700)
	cv2.moveWindow(eyewindow_name, 750, 25)
	cv2.setMouseCallback(eyewindow_name, mouse_callback)

	dot_position = np.array([image_size[0] / 2., image_size[1] / 2.])
	noise = np.array([0., 0.])
	jitter_counter = 0
	i_c = 0  # 8795 - 7000 # 8899 - 7000
	while i_c < num_imgs:

		# quit
		if cv2.getWindowProperty(trackwindow_name, cv2.WND_PROP_VISIBLE) + cv2.getWindowProperty(eyewindow_name, cv2.WND_PROP_VISIBLE) < 2\
			or not controller.isVisible():
			break

		# read & prepare
		frame = cv2.imread(os.path.join(path, img_names[i_c]))
		before = time.perf_counter()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		image_raw = cv2.resize(frame, image_size)
		image_diff_raw = cv2.absdiff(image_raw, img_bg)
		image_diff = cv2.normalize(image_diff_raw, image_diff_raw, 0, 255, cv2.NORM_MINMAX)

		# binarize + erode & dilate (+ mask)
		_, image_bin_body = cv2.threshold(image_diff, threshold_body, 255, cv2.THRESH_BINARY)
		image_bin_body = cv2.erode(image_bin_body, np.ones((5, 5), np.uint8), iterations=1)
		image_bin_body = cv2.dilate(image_bin_body, np.ones((5, 5), np.uint8), iterations=1)
		#image_diff_masked = cv2.bitwise_and(*([image_diff] * 2), mask=image_bin_body)

		# get image mean, using moments
		moments = cv2.moments(image_bin_body)
		if moments['m00'] != 0:
			mean = np.array([moments['m10'] / moments['m00'], moments['m01'] / moments['m00']])
		else:
			mean = np.array([-1., -1.])

		# binarize eyes (+ erode & dilate)
		_, image_bin_eyes = cv2.threshold(image_diff, threshold_eyes, 255, cv2.THRESH_BINARY)  # image_diff_masked
		#image_bin_eyes = cv2.erode(image_bin_eyes, np.ones((1, 1), np.uint8), iterations=1)
		#image_bin_eyes = cv2.dilate(image_bin_eyes, np.ones((3, 3), np.uint8), iterations=1)

		# get centers of eyes, using contours & moments
		ret, _, eye_centers = kmeans(image_bin_eyes, threshold=threshold_eyes)
		eye_centers = eye_centers[np.argsort(eye_centers[:, 0])]


		# dot position
		eyes_center = np.mean(eye_centers.astype(np.float32), axis=0)  # type conversion not necessary, but pleases linter
		if ret < 0:
			dot_position_new = dot_position
		else:
			eye_vector = eye_centers[1] - eye_centers[0]
			eye_vector /= np.linalg.norm(eye_vector)
			eye_vector_orth = np.array([eye_vector[1], -eye_vector[0]])
			orientation = np.sign(np.dot(mean - eyes_center, eye_vector_orth)) * -1
			
			dot_position_new = eyes_center + (eye_vector_orth * distance_y + eye_vector * distance_x) * orientation
			dot_position = dot_position_new
		# jitter
		if jitter_speed > 0:
			jitter_counter += 1
			jitter_speed_log = round(math.log10(jitter_speed) * 50)  # why 50??? but ey, works      # also, maybe just do 10 / speed?
			if jitter_counter >= (100 - jitter_speed_log):
				noise = np.array((random.randrange(-jitter_magnitude, jitter_magnitude + 1), random.randrange(-jitter_magnitude, jitter_magnitude + 1)))
				jitter_counter = 0
		else:
			noise = np.array([0., 0.])
		dot_position_new += noise


		after = time.perf_counter()


		# draw
		image_diff_masked_vis = cv2.cvtColor(image_diff, cv2.COLOR_GRAY2BGR)  # image_diff_masked
		cv2.circle(image_diff_masked_vis, tuple(np.rint(mean).astype(np.intp)), color=(200, 200, 200), radius=5)
		#cv2.line(image_diff_masked_vis, tuple(np.rint(mean - 100. * axis).astype(np.intp)), tuple(np.rint(mean + 100. * axis).astype(np.intp)), color=(200, 200, 200), thickness=2)
		cv2.imshow(trackwindow_name, image_diff_masked_vis)

		image_bin_eyes_vis = cv2.cvtColor(image_diff, cv2.COLOR_GRAY2BGR)
		cv2.circle(image_bin_eyes_vis, tuple(np.rint(mean).astype(np.intp)), color=(200, 200, 200), radius=5)
		if ret >= 0:
			cv2.circle(image_bin_eyes_vis, tuple(np.rint(eye_centers[0]).astype(np.intp)), color=(0, 127, 255), radius=2, thickness=-1)
			cv2.circle(image_bin_eyes_vis, tuple(np.rint(eye_centers[1]).astype(np.intp)), color=(0, 127, 255), radius=2, thickness=-1)
			cv2.line(image_bin_eyes_vis, tuple(np.rint(eye_centers[0]).astype(np.intp)), tuple(np.rint(eye_centers[1]).astype(np.intp)), color=(0, 127, 255), thickness=1)
			cv2.circle(image_bin_eyes_vis, tuple(np.rint(eyes_center).astype(np.intp)), color=dot_color, radius=2, thickness=-1)
			cv2.line(image_bin_eyes_vis, tuple(np.rint(dot_position).astype(np.intp)), tuple(np.rint(eyes_center).astype(np.intp)), color=(0, 127, 255), thickness=1)
		cv2.circle(image_bin_eyes_vis, tuple(np.rint(dot_position).astype(np.intp)), color=dot_color, radius=5, thickness=-1)
		cv2.imshow(eyewindow_name, image_bin_eyes_vis)

		# second screen
		# TODO

		# print
		print(f'{i_c}/{num_imgs} - {1 / (after - before):.2f}fps ' + ('- no eyes found' if ret < 0 else ''))
		cv2.waitKey(16)
		i_c += 1
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
