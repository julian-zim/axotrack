import os
import numpy as np
from screeninfo import get_monitors
from pynput.mouse import Listener
import cv2
import utils


# constants
fps = 60
trackpad_projector_ratio = 1.5
min_zoom = 0.25
max_zoom = 2.
zoom_speed = 0.05


# global variables
image_width = 0.
image_height = 0.
projector_width = 0.
projector_height = 0.
projector_image_ratio = 1.

save = False
move_active = False
start_x, start_y = 0, 0  # always in image space
delta_x_prev, delta_y_prev = 0, 0  # always in image space
delta_x, delta_y = 0, 0  # always in image space
delta_z = 1.


def clamp_delta_xy():
	global delta_x_prev, delta_y_prev, delta_x, delta_y

	# clamp
	min_delta_x = -round(image_width * delta_z) + 1
	max_delta_x = round(projector_width * projector_image_ratio) - 1

	min_delta_y = -round(image_height * delta_z) + 1
	max_delta_y = round(projector_height * projector_image_ratio) - 1

	if not min_delta_x < delta_x < max_delta_x \
		or not min_delta_y < delta_y < max_delta_y:
		delta_x = max(min_delta_x, min(delta_x, max_delta_x))
		delta_y = max(min_delta_y, min(delta_y, max_delta_y))
		return True
	
	return False


def mouse_callback(event, mouse_x, mouse_y, flags, param):
	global move_active
	global start_x, start_y, delta_x_prev, delta_y_prev, delta_x, delta_y

	# trackpad-to-projector-to-image conversion
	mouse_x_imsp = round(mouse_x * trackpad_projector_ratio * projector_image_ratio)
	mouse_y_imsp = round(mouse_y * trackpad_projector_ratio * projector_image_ratio)

	# save
	if event == cv2.EVENT_RBUTTONUP:
		global save
		save = True

	# disable move & apply
	elif event == cv2.EVENT_LBUTTONUP:
		move_active = False
		delta_x_prev, delta_y_prev = delta_x, delta_y

	# enable move & initiate
	elif event == cv2.EVENT_LBUTTONDOWN:
		global start_x, start_y
		start_x, start_y = mouse_x_imsp, mouse_y_imsp
		move_active = True

	# compute total move distance & clamp if necessary
	elif event == cv2.EVENT_MOUSEMOVE:
		if move_active:
			# move
			delta_x_new, delta_y_new = mouse_x_imsp - start_x, mouse_y_imsp - start_y
			delta_x, delta_y = delta_x_prev + delta_x_new, delta_y_prev + delta_y_new
			
			# if delta_x, delta_y out of bounds after move: clamp
			ret = clamp_delta_xy()
			if ret:
				# apply
				delta_x_prev, delta_y_prev = delta_x, delta_y
				# re-initiate, in order to reset mouse position effects
				start_x, start_y = mouse_x_imsp, mouse_y_imsp

	else:
		pass


def on_scroll(wheel_x, wheel_y, wheel_delta_x, wheel_delta_y):

	if not move_active:  # moving & zooming causes bugs
		global min_zoom, max_zoom
		global delta_x_prev, delta_y_prev, delta_x, delta_y
		global delta_z

		delta_z_prev = delta_z
		delta_z *= (1 + zoom_speed) ** wheel_delta_y
		delta_z = max(min_zoom, min(delta_z, max_zoom))

		if delta_z_prev != delta_z:  # if actual zooming happens, adjust the position such that the zoom center is the middle of the image
			
			flat_diff_x = round(image_width * (delta_z - delta_z_prev))
			flat_diff_y = round(image_height * (delta_z - delta_z_prev))

			delta_x = delta_x - flat_diff_x // 2
			delta_y = delta_y - flat_diff_y // 2

			clamp_delta_xy()

			# apply
			delta_x_prev, delta_y_prev = delta_x, delta_y


def sync_projector():
	# get monitors
	monitors = get_monitors()
	if len(monitors) < 2:
		utils.show_error('Cannot find second monitor.')
		quit()
	first_monitor, second_monitor = monitors
	global projector_width, projector_height
	projector_width, projector_height = second_monitor.width, second_monitor.height

	# get image
	image_path = os.path.join('.', 'data', 'background.jpg')
	if not os.path.exists(image_path):
		image_path = utils.choose_file('No background image file could be found at the default location. Please choose one from your PC.')
	if not image_path:
		utils.show_error('Problem choosing file.')
		quit()
	image_orig = cv2.imread(image_path)
	if len(image_orig.shape) != 2:
		image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)

	# pad image_orig to be of the same ratio as the projector (and give it a white border)
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
	
	global image_width, image_height
	image_width, image_height = image.shape[1], image.shape[0]
	global projector_image_ratio
	projector_image_ratio = float(image_height) / float(projector_height)  # should be same as image_width / projector_width

	cv2.imwrite(image_path, image)
	image[0] = np.ones(image_width) * 255
	image[-1] = np.ones(image_width) * 255
	for i in range(1, len(image) - 1):
		image[i][0] = 255
		image[i][-1] = 255

	# setup windows
	proj_window_name = 'Projector'
	cv2.namedWindow(proj_window_name, cv2.WND_PROP_FULLSCREEN)
	cv2.moveWindow(proj_window_name, second_monitor.x, second_monitor.y)
	cv2.setWindowProperty(proj_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

	trackpad_window_name = 'Sync projector with the scene: Left drag to move, mouse wheel to zoom, right click to save.'
	trackpad_width, trackpad_height = round(projector_width / trackpad_projector_ratio), round(projector_height / trackpad_projector_ratio)
	cv2.namedWindow(trackpad_window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(trackpad_window_name, trackpad_width, trackpad_height)
	cv2.moveWindow(trackpad_window_name, first_monitor.x + first_monitor.width // 2 - trackpad_width // 2, first_monitor.y + first_monitor.height // 2 - trackpad_height // 2)
	
	# loop
	frame_delay = round(1000. / fps)
	mouse_listener = Listener(on_scroll=on_scroll)
	mouse_listener.start()
	while True:
		cv2.setMouseCallback(trackpad_window_name, mouse_callback)

		# convert projector shape to image space
		projector_width_imsp = round(projector_width * projector_image_ratio)
		projector_height_imsp = round(projector_height * projector_image_ratio)

		# zoom
		image_zoomed = cv2.resize(image, (round(image_width * delta_z), round(image_height * delta_z)), interpolation=cv2.INTER_LINEAR)

		# pad to make canvas
		image_canvas = np.pad(image_zoomed, ((projector_height_imsp, projector_height_imsp), (projector_width_imsp, projector_width_imsp)))

		# move
		global delta_x, delta_y
		image_moved = image_canvas[
			projector_height_imsp - delta_y:projector_height_imsp + projector_height_imsp - delta_y,
			projector_width_imsp - delta_x:projector_width_imsp + projector_width_imsp - delta_x
		]

		# show image
		projector_image = cv2.resize(image_moved.copy(), (projector_width, projector_height), interpolation=cv2.INTER_LINEAR)
		cv2.imshow(proj_window_name, projector_image)
		# show trackpad
		trackpad_image = cv2.resize(image_moved.copy(), (trackpad_width, trackpad_height), interpolation=cv2.INTER_LINEAR)
		cv2.imshow(trackpad_window_name, trackpad_image)

		# wait
		cv2.waitKey(frame_delay)

		# save image if clicked
		global save
		if save:
			os.makedirs('data', exist_ok=True)
			with open(os.path.join('.', 'data', 'dot_offset.txt'), 'w') as file:
				file.write(f'{delta_x}, {delta_y}\n{delta_z}\n')
			break

		# quit if closed
		if cv2.getWindowProperty(proj_window_name, cv2.WND_PROP_VISIBLE) + cv2.getWindowProperty(trackpad_window_name, cv2.WND_PROP_VISIBLE) < 2:
			break

	cv2.destroyAllWindows()


if __name__ == '__main__':
	sync_projector()
