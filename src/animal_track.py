from sys import argv
from os import getcwd
from os.path import exists, join
from time import sleep, perf_counter
from numpy import array, full, pad, mean, dot, argsort, zeros, ones, sign, rint, uint8, intp
from numpy.random import randint
from numpy.linalg import norm
from screeninfo import get_monitors
from imageio import get_writer
from cv2 import setUseOptimized, setMouseCallback, EVENT_LBUTTONUP, EVENT_RBUTTONUP
from cv2 import namedWindow, moveWindow, resizeWindow, setWindowProperty, getWindowProperty, destroyAllWindows, WINDOW_NORMAL, WINDOW_FULLSCREEN, WND_PROP_FULLSCREEN, WND_PROP_VISIBLE
from cv2 import waitKey, imshow, imread, cvtColor, COLOR_GRAY2BGR, COLOR_BGR2GRAY
from cv2 import resize, flip, line, circle, putText, FONT_HERSHEY_SIMPLEX, LINE_AA
from cv2 import normalize, absdiff, threshold, morphologyEx, moments, findContours, contourArea, NORM_MINMAX, THRESH_BINARY, MORPH_OPEN, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from PyQt5.QtCore import Qt, QTimer, QElapsedTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIntValidator
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QDesktopWidget, QWidget, QLabel, QLineEdit, QSlider, QPushButton, QCheckBox, QSpacerItem, QSizePolicy
from pypylon.pylon import InstantCamera, TlFactory, GrabStrategy_LatestImageOnly, TimeoutHandling_ThrowException
from utils import show_info, show_error, choose_file, save_file, setup_camera


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

	if event == EVENT_LBUTTONUP:
		window_type = (window_type + 1) % len(window_types)

	if event == EVENT_RBUTTONUP:
		window_type = (window_type - 1) % len(window_types)


class ParameterControl(QWidget):
	signal_threshold_body = pyqtSignal(int)
	signal_threshold_eyes = pyqtSignal(int)
	signal_anti_noise_eyes = pyqtSignal(int)
	signal_anti_noise_body = pyqtSignal(int)
	signal_active = pyqtSignal(bool)
	signal_distance_medial = pyqtSignal(int)
	signal_distance_lateral = pyqtSignal(int)
	signal_radius = pyqtSignal(int)
	signal_color = pyqtSignal(tuple)
	signal_jitter_speed = pyqtSignal(int)
	signal_jitter_magnitude_medial = pyqtSignal(int)
	signal_jitter_magnitude_lateral = pyqtSignal(int)
	signal_record = pyqtSignal()
	
	def __init__(self, max_size):
		super().__init__()
		window_width = 320
		label_width = 95
		edit_width = 40

		# threshold body
		self.label_threshold_body = QLabel('Body Threshold:')
		self.label_threshold_body.setFixedWidth(label_width)
	
		self.slider_threshold_body = QSlider(Qt.Horizontal)
		self.slider_threshold_body.setMinimum(0)
		self.slider_threshold_body.setMaximum(255)
		self.slider_threshold_body.setValue(20)
		self.slider_threshold_body.valueChanged.connect(lambda: self.emit_threshold_body(True))

		self.edit_threshold_body = QLineEdit()
		self.edit_threshold_body.setMaximumWidth(edit_width)
		self.edit_threshold_body.setValidator(QIntValidator(0, 255))
		self.edit_threshold_body.setText('20')
		self.edit_threshold_body.editingFinished.connect(lambda: self.emit_threshold_body(False))

		# anti noise body
		self.label_anti_noise_body = QLabel('Body Anti Noise:')
		self.label_anti_noise_body.setFixedWidth(label_width)

		self.slider_anti_noise_body = QSlider(Qt.Horizontal)
		self.slider_anti_noise_body.setMinimum(1)
		self.slider_anti_noise_body.setMaximum(10)
		self.slider_anti_noise_body.setValue(5)
		self.slider_anti_noise_body.valueChanged.connect(lambda: self.emit_anti_noise_body(True))

		self.edit_anti_noise_body = QLineEdit()
		self.edit_anti_noise_body.setMaximumWidth(edit_width)
		self.edit_anti_noise_body.setValidator(QIntValidator(1, 10))
		self.edit_anti_noise_body.setText('5')
		self.edit_anti_noise_body.editingFinished.connect(lambda: self.emit_anti_noise_body(False))

		# threshold eyes
		self.label_threshold_eyes = QLabel('Eyes Threshold:')
		self.label_threshold_eyes.setFixedWidth(label_width)

		self.slider_threshold_eyes = QSlider(Qt.Horizontal)
		self.slider_threshold_eyes.setMinimum(0)
		self.slider_threshold_eyes.setMaximum(255)
		self.slider_threshold_eyes.setValue(200)
		self.slider_threshold_eyes.valueChanged.connect(lambda: self.emit_threshold_eyes(True))

		self.edit_threshold_eyes = QLineEdit()
		self.edit_threshold_eyes.setMaximumWidth(edit_width)
		self.edit_threshold_eyes.setValidator(QIntValidator(0, 255))
		self.edit_threshold_eyes.setText('200')
		self.edit_threshold_eyes.editingFinished.connect(lambda: self.emit_threshold_eyes(False))

		# anti noise eyes
		self.label_anti_noise_eyes = QLabel('Eyes Anti Noise:')
		self.label_anti_noise_eyes.setFixedWidth(label_width)

		self.slider_anti_noise_eyes = QSlider(Qt.Horizontal)
		self.slider_anti_noise_eyes.setMinimum(1)
		self.slider_anti_noise_eyes.setMaximum(10)
		self.slider_anti_noise_eyes.setValue(1)
		self.slider_anti_noise_eyes.valueChanged.connect(lambda: self.emit_anti_noise_eyes(True))

		self.edit_anti_noise_eyes = QLineEdit()
		self.edit_anti_noise_eyes.setMaximumWidth(edit_width)
		self.edit_anti_noise_eyes.setValidator(QIntValidator(1, 10))
		self.edit_anti_noise_eyes.setText('1')
		self.edit_anti_noise_eyes.editingFinished.connect(lambda: self.emit_anti_noise_eyes(False))

		# dot distance medial
		self.label_distance_medial = QLabel('Medial Distance:')
		self.label_distance_medial.setFixedWidth(label_width)

		self.slider_distance_medial = QSlider(Qt.Horizontal)
		self.slider_distance_medial.setMinimum(-max_size)
		self.slider_distance_medial.setMaximum(max_size)
		self.slider_distance_medial.setValue(60)
		self.slider_distance_medial.valueChanged.connect(lambda: self.emit_distance_medial(True))

		self.edit_distance_medial = QLineEdit()
		self.edit_distance_medial.setMaximumWidth(edit_width)
		self.edit_distance_medial.setValidator(QIntValidator(-max_size, max_size))
		self.edit_distance_medial.setText('60')
		self.edit_distance_medial.editingFinished.connect(lambda: self.emit_distance_medial(False))

		# dot distance lateral
		self.label_distance_lateral = QLabel('Lateral Distance:')
		self.label_distance_lateral.setFixedWidth(label_width)

		self.slider_distance_lateral = QSlider(Qt.Horizontal)
		self.slider_distance_lateral.setMinimum(-max_size)
		self.slider_distance_lateral.setMaximum(max_size)
		self.slider_distance_lateral.setValue(0)
		self.slider_distance_lateral.valueChanged.connect(lambda: self.emit_distance_lateral(True))

		self.edit_distance_lateral = QLineEdit()
		self.edit_distance_lateral.setMaximumWidth(edit_width)
		self.edit_distance_lateral.setValidator(QIntValidator(-max_size, max_size))
		self.edit_distance_lateral.setText('0')
		self.edit_distance_lateral.editingFinished.connect(lambda: self.emit_distance_lateral(False))

		# dot size
		self.label_radius = QLabel('Size:')
		self.label_radius.setFixedWidth(label_width)

		self.slider_radius = QSlider(Qt.Horizontal)
		self.slider_radius.setMinimum(0)
		self.slider_radius.setMaximum(max_size // 2)
		self.slider_radius.setValue(5)
		self.slider_radius.valueChanged.connect(lambda: self.emit_radius(True))

		self.edit_radius = QLineEdit()
		self.edit_radius.setMaximumWidth(edit_width)
		self.edit_radius.setValidator(QIntValidator(0, max_size))
		self.edit_radius.setText('5')
		self.edit_radius.editingFinished.connect(lambda: self.emit_radius(False))

		# dot color
		self.label_hue = QLabel('Color:')
		self.label_hue.setFixedWidth(label_width)

		self.slider_hue = QSlider(Qt.Horizontal)
		self.slider_hue.setMinimum(0)
		self.slider_hue.setMaximum(255)
		self.slider_hue.setValue(255)
		self.slider_hue.valueChanged.connect(lambda: self.emit_color(True))

		self.edit_hue = QLineEdit()
		self.edit_hue.setMaximumWidth(edit_width)
		self.edit_hue.setValidator(QIntValidator(0, 255))
		self.edit_hue.setText('255')
		self.edit_hue.editingFinished.connect(lambda: self.emit_color(False))

		# dot brightness
		self.label_brightness = QLabel('Brightness:')
		self.label_brightness.setFixedWidth(label_width)

		self.slider_brightness = QSlider(Qt.Horizontal)
		self.slider_brightness.setMinimum(0)
		self.slider_brightness.setMaximum(511)
		self.slider_brightness.setValue(255)
		self.slider_brightness.valueChanged.connect(lambda: self.emit_color(True))

		self.edit_brightness = QLineEdit()
		self.edit_brightness.setMaximumWidth(edit_width)
		self.edit_brightness.setValidator(QIntValidator(0, 511))
		self.edit_brightness.setText('255')
		self.edit_brightness.editingFinished.connect(lambda: self.emit_color(False))

		# dot jitter speed
		self.label_jitter_speed = QLabel('Jitter Speed:')
		self.label_jitter_speed.setFixedWidth(label_width) 

		self.slider_jitter_speed = QSlider(Qt.Horizontal)
		self.slider_jitter_speed.setMinimum(0)
		self.slider_jitter_speed.setMaximum(100)
		self.slider_jitter_speed.setValue(90)
		self.slider_jitter_speed.valueChanged.connect(lambda: self.emit_jitter_speed(True))

		self.edit_jitter_speed = QLineEdit()
		self.edit_jitter_speed.setMaximumWidth(edit_width)
		self.edit_jitter_speed.setValidator(QIntValidator(0, 100))
		self.edit_jitter_speed.setText('90')
		self.edit_jitter_speed.editingFinished.connect(lambda: self.emit_jitter_speed(False))

		# dot jitter strength medial
		self.label_jitter_magnitude_medial = QLabel('Medial Jitter:')
		self.label_jitter_magnitude_medial.setFixedWidth(label_width)

		self.slider_jitter_magnitude_medial = QSlider(Qt.Horizontal)
		self.slider_jitter_magnitude_medial.setMinimum(0)
		self.slider_jitter_magnitude_medial.setMaximum(max_size)
		self.slider_jitter_magnitude_medial.setValue(5)
		self.slider_jitter_magnitude_medial.valueChanged.connect(lambda: self.emit_jitter_magnitude_medial(True))

		self.edit_jitter_magnitude_medial = QLineEdit()
		self.edit_jitter_magnitude_medial.setMaximumWidth(edit_width)
		self.edit_jitter_magnitude_medial.setValidator(QIntValidator(0, max_size))
		self.edit_jitter_magnitude_medial.setText('5')
		self.edit_jitter_magnitude_medial.editingFinished.connect(lambda: self.emit_jitter_magnitude_medial(False))

		# dot jitter strength lateral
		self.label_jitter_magnitude_lateral = QLabel('Lateral Jitter:')
		self.label_jitter_magnitude_lateral.setFixedWidth(label_width)

		self.slider_jitter_magnitude_lateral = QSlider(Qt.Horizontal)
		self.slider_jitter_magnitude_lateral.setMinimum(0)
		self.slider_jitter_magnitude_lateral.setMaximum(max_size)
		self.slider_jitter_magnitude_lateral.setValue(10)
		self.slider_jitter_magnitude_lateral.valueChanged.connect(lambda: self.emit_jitter_magnitude_lateral(True))

		self.edit_jitter_magnitude_lateral = QLineEdit()
		self.edit_jitter_magnitude_lateral.setMaximumWidth(edit_width)
		self.edit_jitter_magnitude_lateral.setValidator(QIntValidator(0, max_size))
		self.edit_jitter_magnitude_lateral.setText('10')
		self.edit_jitter_magnitude_lateral.editingFinished.connect(lambda: self.emit_jitter_magnitude_lateral(False))

		# dot active
		self.label_active = QLabel('Dot Active:')
		self.label_active.setFixedWidth(label_width)
		self.label_active.setMinimumHeight(20)
	
		self.checkbox_active = QCheckBox()
		self.checkbox_active.setChecked(False)
		self.checkbox_active.stateChanged.connect(lambda: self.signal_active.emit(self.checkbox_active.isChecked()))

		# button record
		self.button_record = QPushButton('Start Recording')
		self.button_record.clicked.connect(self.emit_record)

		# timer
		self.timer_ms = 0
		self.edit_timer = QLineEdit("00:00:000")
		self.edit_timer.setReadOnly(True)
		self.edit_timer.setAlignment(Qt.AlignCenter)
		self.timer_elapsed = QElapsedTimer()
		self.timer = QTimer()
		self.timer.timeout.connect(self.update_timer)

		# button quit
		self.button_quit = QPushButton('Save video and quit')
		self.button_quit.clicked.connect(self.close)

		# region layouts
		layout_threshold_body = QHBoxLayout()
		layout_threshold_body.addWidget(self.label_threshold_body)
		layout_threshold_body.addWidget(self.slider_threshold_body)
		layout_threshold_body.addWidget(self.edit_threshold_body)
		layout_threshold_body.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_anti_noise_body = QHBoxLayout()
		layout_anti_noise_body.addWidget(self.label_anti_noise_body)
		layout_anti_noise_body.addWidget(self.slider_anti_noise_body)
		layout_anti_noise_body.addWidget(self.edit_anti_noise_body)
		layout_anti_noise_body.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_threshold_eyes = QHBoxLayout()
		layout_threshold_eyes.addWidget(self.label_threshold_eyes)
		layout_threshold_eyes.addWidget(self.slider_threshold_eyes)
		layout_threshold_eyes.addWidget(self.edit_threshold_eyes)
		layout_threshold_eyes.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_anti_noise_eyes = QHBoxLayout()
		layout_anti_noise_eyes.addWidget(self.label_anti_noise_eyes)
		layout_anti_noise_eyes.addWidget(self.slider_anti_noise_eyes)
		layout_anti_noise_eyes.addWidget(self.edit_anti_noise_eyes)
		layout_anti_noise_eyes.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_distance_medial = QHBoxLayout()
		layout_distance_medial.addWidget(self.label_distance_medial)
		layout_distance_medial.addWidget(self.slider_distance_medial)
		layout_distance_medial.addWidget(self.edit_distance_medial)
		layout_distance_medial.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_distance_lateral = QHBoxLayout()
		layout_distance_lateral.addWidget(self.label_distance_lateral)
		layout_distance_lateral.addWidget(self.slider_distance_lateral)
		layout_distance_lateral.addWidget(self.edit_distance_lateral)
		layout_distance_lateral.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_radius = QHBoxLayout()
		layout_radius.addWidget(self.label_radius)
		layout_radius.addWidget(self.slider_radius)
		layout_radius.addWidget(self.edit_radius)
		layout_radius.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_hue = QHBoxLayout()
		layout_hue.addWidget(self.label_hue)
		layout_hue.addWidget(self.slider_hue)
		layout_hue.addWidget(self.edit_hue)
		layout_hue.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_brightness = QHBoxLayout()
		layout_brightness.addWidget(self.label_brightness)
		layout_brightness.addWidget(self.slider_brightness)
		layout_brightness.addWidget(self.edit_brightness)
		layout_brightness.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_jitter_speed = QHBoxLayout()
		layout_jitter_speed.addWidget(self.label_jitter_speed)
		layout_jitter_speed.addWidget(self.slider_jitter_speed)
		layout_jitter_speed.addWidget(self.edit_jitter_speed)
		layout_jitter_speed.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_jitter_magnitude_medial = QHBoxLayout()
		layout_jitter_magnitude_medial.addWidget(self.label_jitter_magnitude_medial)
		layout_jitter_magnitude_medial.addWidget(self.slider_jitter_magnitude_medial)
		layout_jitter_magnitude_medial.addWidget(self.edit_jitter_magnitude_medial)
		layout_jitter_magnitude_medial.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_jitter_magnitude_lateral = QHBoxLayout()
		layout_jitter_magnitude_lateral.addWidget(self.label_jitter_magnitude_lateral)
		layout_jitter_magnitude_lateral.addWidget(self.slider_jitter_magnitude_lateral)
		layout_jitter_magnitude_lateral.addWidget(self.edit_jitter_magnitude_lateral)
		layout_jitter_magnitude_lateral.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_active = QHBoxLayout()
		layout_active.addWidget(self.label_active)
		layout_active.addWidget(self.checkbox_active)
		layout_active.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

		layout_main = QVBoxLayout()
		layout_main.addLayout(layout_threshold_body)
		layout_main.addLayout(layout_anti_noise_body)
		layout_main.addLayout(layout_threshold_eyes)
		layout_main.addLayout(layout_anti_noise_eyes)
		layout_main.addLayout(layout_distance_medial)
		layout_main.addLayout(layout_distance_lateral)
		layout_main.addLayout(layout_radius)
		layout_main.addLayout(layout_hue)
		layout_main.addLayout(layout_brightness)
		layout_main.addLayout(layout_jitter_speed)
		layout_main.addLayout(layout_jitter_magnitude_medial)
		layout_main.addLayout(layout_jitter_magnitude_lateral)
		layout_main.addLayout(layout_active)
		layout_main.addItem(QSpacerItem(0, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
		layout_main.addWidget(self.button_record)
		#layout_main.addWidget(self.edit_timer)
		layout_main.addWidget(self.button_quit)
		self.setLayout(layout_main)
		# endregion
	
		# window
		self.setWindowTitle("Control Panel")
		self.setMinimumWidth(window_width)
		self.adjustSize()
		_, _, screen_width, screen_height = QDesktopWidget().screenGeometry().getRect()
		_, _, width, height = self.geometry().getRect()
		self.move(screen_width * 3 // 4 - width // 2, screen_height // 2 - height // 2)
	

	def emit_threshold_body(self, from_slider):
		if from_slider:
			self.edit_threshold_body.setText(str(self.slider_threshold_body.value()))
		else:
			self.slider_threshold_body.setValue(int(self.edit_threshold_body.text()))
		self.signal_threshold_body.emit(self.slider_threshold_body.value())

	def emit_anti_noise_body(self, from_slider):
		if from_slider:
			self.edit_anti_noise_body.setText(str(self.slider_anti_noise_body.value()))
		else:
			self.slider_anti_noise_body.setValue(int(self.edit_anti_noise_body.text()))
		self.signal_anti_noise_body.emit(self.slider_anti_noise_body.value())

	def emit_threshold_eyes(self, from_slider):
		if from_slider:
			self.edit_threshold_eyes.setText(str(self.slider_threshold_eyes.value()))
		else:
			self.slider_threshold_eyes.setValue(int(self.edit_threshold_eyes.text()))
		self.signal_threshold_eyes.emit(self.slider_threshold_eyes.value())

	def emit_anti_noise_eyes(self, from_slider):
		if from_slider:
			self.edit_anti_noise_eyes.setText(str(self.slider_anti_noise_eyes.value()))
		else:
			self.slider_anti_noise_eyes.setValue(int(self.edit_anti_noise_eyes.text()))
		self.signal_anti_noise_eyes.emit(self.slider_anti_noise_eyes.value())

	def emit_distance_medial(self, from_slider):
		if from_slider:
			self.edit_distance_medial.setText(str(self.slider_distance_medial.value()))
		else:
			self.slider_distance_medial.setValue(int(self.edit_distance_medial.text()))
		self.signal_distance_medial.emit(self.slider_distance_medial.value())

	def emit_distance_lateral(self, from_slider):
		if from_slider:
			self.edit_distance_lateral.setText(str(self.slider_distance_lateral.value()))
		else:
			self.slider_distance_lateral.setValue(int(self.edit_distance_lateral.text()))
		self.signal_distance_lateral.emit(self.slider_distance_lateral.value())

	def emit_radius(self, from_slider):
		if from_slider:
			self.edit_radius.setText(str(self.slider_radius.value()))
		else:
			self.slider_radius.setValue(int(self.edit_radius.text()))
		self.signal_radius.emit(self.slider_radius.value())

	def emit_color(self, from_slider):
		if from_slider:
			self.edit_hue.setText(str(self.slider_hue.value()))
			self.edit_brightness.setText(str(self.slider_brightness.value()))
		else:
			self.slider_hue.setValue(int(self.edit_hue.text()))
			self.slider_brightness.setValue(int(self.edit_brightness.text()))
		color = QColor()
		color.setHsv(self.slider_hue.value(), 255 - max(0, self.slider_brightness.value() - 256), min(self.slider_brightness.value(), 255))
		self.signal_color.emit(color.getRgb()[:3])

	def emit_jitter_speed(self, from_slider):
		if from_slider:
			self.edit_jitter_speed.setText(str(self.slider_jitter_speed.value()))
		else:
			self.slider_jitter_speed.setValue(int(self.edit_jitter_speed.text()))
		self.signal_jitter_speed.emit(self.slider_jitter_speed.value())

	def emit_jitter_magnitude_medial(self, from_slider):
		if from_slider:
			self.edit_jitter_magnitude_medial.setText(str(self.slider_jitter_magnitude_medial.value()))
		else:
			self.slider_jitter_magnitude_medial.setValue(int(self.edit_jitter_magnitude_medial.text()))
		self.signal_jitter_magnitude_medial.emit(self.slider_jitter_magnitude_medial.value())

	def emit_jitter_magnitude_lateral(self, from_slider):
		if from_slider:
			self.edit_jitter_magnitude_lateral.setText(str(self.slider_jitter_magnitude_lateral.value()))
		else:
			self.slider_jitter_magnitude_lateral.setValue(int(self.edit_jitter_magnitude_lateral.text()))
		self.signal_jitter_magnitude_lateral.emit(self.slider_jitter_magnitude_lateral.value())

	def emit_record(self):
		self.button_record.setEnabled(False)
		self.button_record.setText('Recording...')
		#self.timer_elapsed.start()
		#self.timer.start(10)
		self.signal_record.emit()

	def emit_all(self):
		self.emit_threshold_body(True)
		self.emit_threshold_eyes(True)
		self.emit_anti_noise_body(True)
		self.emit_anti_noise_eyes(True)
		self.signal_active.emit(self.checkbox_active.isChecked())
		self.emit_distance_medial(True)
		self.emit_distance_lateral(True)
		self.emit_radius(True)
		self.emit_color(True)
		self.emit_jitter_speed(True)
		self.emit_jitter_magnitude_medial(True)
		self.emit_jitter_magnitude_lateral(True)
	
	def update_timer(self):
		time = self.timer_elapsed.elapsed()
		minutes, remainder = divmod(time, 60000)
		seconds, milliseconds = divmod(remainder, 1000)
		time = f'{minutes:02d}:{seconds:02d}:{milliseconds:03d}'
		self.edit_timer.setText(time)

	def closeEvent(self, event):
		QApplication.exit(0)


def main():
	max_quality_size = 1024
	max_jitter_delay_s = 3
	setUseOptimized(True)


	# background setup
	dirpath = join(getcwd(), 'data')

	filepath_background = join(dirpath, f'background.jpg')
	if not exists(filepath_background):
		filepath_background = choose_file('No background image was found. Please specify one manually.')
	image_bg = imread(filepath_background)
	if image_bg is None:
		show_error('Cannot read background image.')
		return -1
	image_bg = cvtColor(image_bg, COLOR_BGR2GRAY)
	if image_bg.shape[0] != image_bg.shape[1]:
		show_error('Background image must be square.')
		return -1
	image_square_size = min(image_bg.shape[0], max_quality_size)
	image_bg = resize(image_bg, (image_square_size, image_square_size))

	filepath_backlight = join(getcwd(), 'data', f'backlight.txt')
	if not exists(filepath_backlight):
		show_info('No backlight file was found. Defaulting to maximum brightness.')
		background_brightness = 255
	else:
		with open(filepath_backlight, 'r') as file:
			background_brightness = int(file.read())
	background_color = (background_brightness, background_brightness, background_brightness)


	# camera setup
	try:
		camera = InstantCamera(TlFactory.GetInstance().CreateFirstDevice())
	except:
		show_error('Could not access the camera. Make sure all other software that accesses it is closed.')
		return -1
	camera.Open()
	if not camera.IsOpen():
		show_error('Cannot find camera.')
		return -1
	setup_camera(camera)


	# window setup
	monitors = get_monitors()

	trackwindow_name = 'Tracking Window (click to change view)'
	namedWindow(trackwindow_name, WINDOW_NORMAL)
	resizeWindow(trackwindow_name, monitors[0].height * 8 // 10, monitors[0].height * 8 // 10)
	moveWindow(trackwindow_name, monitors[0].x + monitors[0].height * 1 // 10, monitors[0].y + monitors[0].height * 1 // 10)
	setMouseCallback(trackwindow_name, mouse_callback)

	if len(monitors) > 1:
		projwindow_name = 'Projection Window'
		namedWindow(projwindow_name, WINDOW_NORMAL)
		moveWindow(projwindow_name, monitors[1].x, monitors[1].y)
		resizeWindow(projwindow_name, monitors[1].width, monitors[1].height)
		setWindowProperty(projwindow_name, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN)
		image_proj = full((image_square_size, image_square_size, 3), background_brightness, dtype=uint8)
		pad_width = round((image_square_size * monitors[1].width / monitors[1].height - image_square_size) / 2)
		image_proj = pad(image_proj, ((0, 0), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
	else:
		show_info('No second screen detected!')


	# output setup
	outfile_path = save_file('Choose a video output file name & location.', getcwd(), [('MP4', '.mp4')], '.mp4', 'output.mp4')
	video_writer = get_writer(outfile_path, fps=30)


	# parameter dialog setup
	threshold_body = -1
	threshold_eyes = -1
	antinoise_body = -1
	antinoise_eyes = -1
	distance_y = -1
	distance_x = -1
	radius = -1
	color = (-1, -1, -1)
	jitter_speed = -1
	jitter_delay = -1
	jitter_magnitude_y = (-1, -1)
	jitter_magnitude_x = (-1, -1)
	active = None
	record = False
	def update_threshold_body(threshold_body_new):
		nonlocal threshold_body
		threshold_body = threshold_body_new
	def update_threshold_eyes(threshold_eyes_new):
		nonlocal threshold_eyes
		threshold_eyes = threshold_eyes_new
	def update_antinoise_body(antinoise_body_new):
		nonlocal antinoise_body
		antinoise_body = antinoise_body_new
	def update_antinoise_eyes(antinoise_eyes_new):
		nonlocal antinoise_eyes
		antinoise_eyes = antinoise_eyes_new
	def update_distance_y(distance_y_new):
		nonlocal distance_y
		distance_y = distance_y_new
	def update_distance_x(distance_x_new):
		nonlocal distance_x
		distance_x = distance_x_new
	def update_radius(radius_new):
		nonlocal radius
		radius = radius_new
	def update_color(color_new):
		nonlocal color
		color = color_new
	def update_jitter_speed(jitter_speed_new):
		nonlocal jitter_speed, jitter_delay
		jitter_speed = jitter_speed_new
		jitter_delay = round(((100 - jitter_speed_new) / 100)**2 * 30 * max_jitter_delay_s)
	def update_jitter_magnitude_y(jitter_magnitude_y_new):
		nonlocal jitter_magnitude_y
		jitter_magnitude_y = (-jitter_magnitude_y_new, jitter_magnitude_y_new + 1)
	def update_jitter_magnitude_x(jitter_magnitude_x_new):
		nonlocal jitter_magnitude_x
		jitter_magnitude_x = (-jitter_magnitude_x_new, jitter_magnitude_x_new + 1)
	def update_active(active_new):
		nonlocal active
		active = active_new
	def start_record():
		nonlocal record
		record = True
	app = QApplication(argv)
	controller = ParameterControl(max_quality_size)
	controller.signal_threshold_body.connect(update_threshold_body)
	controller.signal_threshold_eyes.connect(update_threshold_eyes)
	controller.signal_anti_noise_body.connect(update_antinoise_body)
	controller.signal_anti_noise_eyes.connect(update_antinoise_eyes)
	controller.signal_distance_medial.connect(update_distance_y)
	controller.signal_distance_lateral.connect(update_distance_x)
	controller.signal_radius.connect(update_radius)
	controller.signal_color.connect(update_color)
	controller.signal_jitter_speed.connect(update_jitter_speed)
	controller.signal_jitter_magnitude_medial.connect(update_jitter_magnitude_y)
	controller.signal_jitter_magnitude_lateral.connect(update_jitter_magnitude_x)
	controller.signal_active.connect(update_active)
	controller.signal_record.connect(start_record)
	controller.emit_all()  # initialize variables (except record)
	controller.show()


	# main loop
	max_time_ms = 1000 / 30 - 1

	dot_position_prev = full(2, -1.)
	noise = zeros(2)
	jitter_counter = 0

	camera.StartGrabbing(GrabStrategy_LatestImageOnly)
	while camera.IsGrabbing() and controller.isVisible() and getWindowProperty(trackwindow_name, WND_PROP_VISIBLE):
		before = perf_counter()


		# read
		grabResult = camera.RetrieveResult(1000, TimeoutHandling_ThrowException)
		if not grabResult.GrabSucceeded(): break
		frame = flip(grabResult.Array, -1)
		grabResult.Release()
		if len(frame.shape) != 2:
			frame = cvtColor(frame, COLOR_BGR2GRAY)


		# preprocess
		image_raw = resize(frame, (image_square_size, image_square_size))  # should be square already; however, efficiency can be tuned by changing max_quality_size
		image_diff_raw = absdiff(image_raw, image_bg)
		image_diff = normalize(image_diff_raw, None, 0, 255, NORM_MINMAX)  #type: ignore


		# body
		_, image_bin_body = threshold(image_diff, threshold_body, 255, THRESH_BINARY)
		if antinoise_body > 1:
			image_bin_body = morphologyEx(image_bin_body, MORPH_OPEN, ones((antinoise_body, antinoise_body), uint8), iterations=1)   # erode & dilate

		body_centroid = full(2, -1.)
		image_moments = moments(image_bin_body)
		if image_moments['m00'] > 0:
			body_centroid = array([image_moments['m10'] / image_moments['m00'], image_moments['m01'] / image_moments['m00']])


		# eyes
		_, image_bin_eyes = threshold(image_diff, threshold_eyes, 255, THRESH_BINARY)
		if antinoise_eyes > 1:
			image_bin_eyes = morphologyEx(image_bin_eyes, MORPH_OPEN, ones((antinoise_eyes, antinoise_eyes), uint8), iterations=1)   # erode & dilate

		contours, _ = findContours(image_bin_eyes, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
		#if len(contours) > 2:
			#print('Warning! More than two eyes detected?')

		eyes_centroids = full((2, 2), -1.)
		if len(contours) >= 2:
			contours_sorted = sorted(contours, key=contourArea, reverse=True)[:2]  # two largest contours (hopefully the eyes)

			image_moments = (moments(contours_sorted[0]), moments(contours_sorted[1]))
			if image_moments[0]['m00'] > 0:
				eyes_centroids[0] = (array([image_moments[0]['m10'] / image_moments[0]['m00'], image_moments[0]['m01'] / image_moments[0]['m00']]))
			if image_moments[1]['m00'] > 0:
				eyes_centroids[1] = (array([image_moments[1]['m10'] / image_moments[1]['m00'], image_moments[1]['m01'] / image_moments[1]['m00']]))

		eyes_found = not any((eyes_centroids < 0.).flatten())


		# dot
		if eyes_found:  # in case of eyes detected, use eye orientation for dot position and jitter orientation
			eyes_centroids = eyes_centroids[argsort(eyes_centroids[:, 0])]
			eyes_center = mean(eyes_centroids, axis=0)

			lateral_vector = eyes_centroids[1] - eyes_centroids[0]
			lateral_vector_norm = norm(lateral_vector)
			if lateral_vector_norm > 0:
				lateral_vector /= lateral_vector_norm
			medial_vector = array([lateral_vector[1], -lateral_vector[0]])
			
			orientation = sign(dot(eyes_center - body_centroid, medial_vector))
			
			dot_position = eyes_center + (medial_vector * distance_y + lateral_vector * distance_x) * orientation
			dot_position_prev = dot_position

		else:  # in case of eyes not detected, keep last dot position and use relative orientation towards it as jitter orientation
			dot_position = dot_position_prev.copy()
			medial_vector = dot_position - body_centroid
			medial_vector_norm = norm(medial_vector)
			if medial_vector_norm > 0:
				medial_vector /= medial_vector_norm
			lateral_vector = array([medial_vector[1], -medial_vector[0]])

		# jitter
		if jitter_speed > 0:
			if jitter_counter >= jitter_delay:
				noise = medial_vector * randint(*jitter_magnitude_y) + lateral_vector * randint(*jitter_magnitude_x)
				jitter_counter = 0
			else:
				jitter_counter += 1
			dot_position += noise


		# draw
		match window_type:
			case 0:
				image_vis = cvtColor(image_raw, COLOR_GRAY2BGR)
			case 1:
				image_vis = cvtColor(image_diff_raw, COLOR_GRAY2BGR)
			case 2:
				image_vis = cvtColor(image_diff, COLOR_GRAY2BGR)
			case 3:
				image_vis = cvtColor(image_bin_body, COLOR_GRAY2BGR)
			case 4:
				image_vis = cvtColor(image_bin_eyes, COLOR_GRAY2BGR)
			case _:
				image_vis = cvtColor(image_diff, COLOR_GRAY2BGR)

		# main screen
		circle(image_vis, tuple(rint(body_centroid).astype(intp)), color=(200, 200, 200), radius=5)  # body centroid
		if eyes_found:
			circle(image_vis, tuple(rint(eyes_centroids[0]).astype(intp)), color=(0, 127, 255), radius=2, thickness=-1)  # eye 1 centroid
			circle(image_vis, tuple(rint(eyes_centroids[1]).astype(intp)), color=(0, 127, 255), radius=2, thickness=-1)  # eye 2 centroid
			circle(image_vis, tuple(rint(eyes_center).astype(intp)), color=(0, 127, 255), radius=2, thickness=-1)  # eyes center
			line(image_vis, tuple(rint(eyes_centroids[0]).astype(intp)), tuple(rint(eyes_centroids[1]).astype(intp)), color=(0, 127, 255), thickness=1)  # eyes vector
			line(image_vis, tuple(rint(dot_position).astype(intp)), tuple(rint(eyes_center).astype(intp)), color=(0, 127, 255), thickness=1)  # eyes vector orth
		circle(image_vis, tuple(rint(dot_position).astype(intp)), color=color, radius=radius, thickness=-1)  # dot

		putText(image_vis, f'{str(window_type)}: {window_types[window_type]}', (10, 25), FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, LINE_AA)
		imshow(trackwindow_name, image_vis)

		# projector
		if len(monitors) > 1:
			if active:
				circle(image_proj, tuple(rint((dot_position[0] + pad_width, dot_position[1])).astype(intp)), color=color, radius=radius, thickness=-1)  # draw dot
			imshow(projwindow_name, image_proj)
			if active:
				circle(image_proj, tuple(rint((dot_position[0] + pad_width, dot_position[1])).astype(intp)), color=background_color, radius=radius, thickness=-1)  # remove dot again for next draw


		# save
		if record:
			image_dot = cvtColor(image_raw, COLOR_GRAY2BGR)
			if active:
				circle(image_dot, tuple(rint((dot_position[0], dot_position[1])).astype(intp)), color=color[::-1], radius=radius, thickness=-1)  # draw dot
			video_writer.append_data(image_dot)


		# end
		after = perf_counter()
		passed_time_ms = (after - before) * 1000
		diff_time_ms = max_time_ms - passed_time_ms
		if diff_time_ms >= 0:  # wait for the rest of the frame before writing onto video to ensure 30 fps
			diff_time_ms_floor = int(diff_time_ms)
			waitKey(1 + diff_time_ms_floor)
			sleep((diff_time_ms - diff_time_ms_floor) / 1000)
		else:  # in this case, the calculations exceeded ~33ms. Video will be less than 30 fps. TODO: implement frame skipping
			waitKey(1)
			#print('WARNING: Frame took too long to compute.')
		#print(f'{1 / (after - before):.2f}fps - {(after - before)*1000:.2f}ms ' + ('' if eyes_found else '- no eyes found'))


	# quit
	camera.StopGrabbing()
	camera.Close()
	video_writer.close()
	destroyAllWindows()
	controller.close()
	app.quit()

	if record: show_info(f'Video saved to\n\n"{outfile_path}".')

	return 0


if __name__ == '__main__':
	print('PROGRAM STARTED')
	main()
