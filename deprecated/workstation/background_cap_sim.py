import os
from screeninfo import get_monitors
import cv2
# placeholder
import utils
import numpy as np

# constants
max_fps = 60.
window_screen_ratio = 1.25

# global variables
left_click = False


def mouse_callback(event, x, y, flags, param):
	global left_click

	if event == cv2.EVENT_RBUTTONUP:
		left_click = True


def capture_background():
	# get monitors
	monitors = get_monitors()
	#if len(monitors) < 2:
	#	utils.show_error('Cannot find second monitor.')
	#	quit()
	first_monitor = monitors[0]  # first_monitor, second_monitor = monitors

	# setup camera
	# placeholder
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		print('Cannot find camera.')
		return
	# placeholder
	camera_ratio = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) / float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# setup windows
	#proj_window_name = 'Projector'
	#proj_frame = np.ones((second_monitor.height, second_monitor.width))
	#cv2.namedWindow(proj_window_name, cv2.WND_PROP_FULLSCREEN)
	#cv2.moveWindow(proj_window_name, second_monitor.x, second_monitor.y)
	#cv2.setWindowProperty(proj_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	cap_window_name = 'Capture background: Right click anywhere to take a photo.'
	cap_window_height = round(first_monitor.height / window_screen_ratio)
	cap_window_width = round(cap_window_height * camera_ratio)
	cv2.namedWindow(cap_window_name , cv2.WINDOW_KEEPRATIO)
	cv2.resizeWindow(cap_window_name, cap_window_width, cap_window_height)
	cv2.moveWindow(cap_window_name, first_monitor.x + first_monitor.width // 2 - cap_window_width // 2,
								first_monitor.y + first_monitor.height // 2 - cap_window_height // 2)

	# loop
	frame_delay = round(1000. / max_fps)
	# placeholder
	while True:
		ret, frame = cap.read()
		if not ret: break
		# placeholder
		if len(frame.shape) != 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#cv2.imshow(proj_window_name, proj_frame)
		cv2.imshow(cap_window_name, frame)
		cv2.setMouseCallback(cap_window_name, mouse_callback)

		cv2.waitKey(frame_delay)
		
		if cv2.getWindowProperty(cap_window_name, cv2.WND_PROP_VISIBLE) < 1:
			break

		if left_click:
			os.makedirs('data', exist_ok=True)
			cv2.imwrite(os.path.join('.', 'data', 'background.jpg'), frame)
			break
	
	# destruct
	cap.release()
	# placeholder
	cv2.destroyAllWindows()
		

if __name__ == '__main__':
	capture_background()
