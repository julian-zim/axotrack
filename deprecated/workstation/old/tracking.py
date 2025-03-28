import numpy as np
import cv2


# use camshift for roi and rotation
# use use ellipse fitting for orientation


def get_rois(image, image_bg, max_track_window_size, track_window, term_crit):
	image_diff = cv2.subtract(image, image_bg)  # TODO: black on white or white on black?

	# binarize
	cv2.normalize(image_diff, image_diff, 0, 255, cv2.NORM_MINMAX)

	
	# cam shift tracking
	rot_window, track_window = cv2.CamShift(image_diff, track_window, term_crit)
	track_w_max, track_h_max = max_track_window_size
	track_x, track_y, track_w, track_h = track_window
	(rot_cx, rot_cy), (rot_w, rot_h), angle = rot_window  # angle of the width axis wrt to the x-axis

	# limit track window size
	if track_w > track_w_max:
		track_w = track_w_max
		track_window = track_x, track_y, track_w_max, track_h
	if track_h > track_h_max:
		track_h = track_h_max
		track_window = track_x, track_y, track_w, track_h_max

	# fix angle & shape
	if rot_w > rot_h:  
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

		
	(track_x, track_y, track_w, track_h) = track_window

	return rot_window, track_window



def roi_backrotation(image, angle):
	height, width = image.shape

	max_dim = max([width, height])
	diff_w = (max_dim - width) // 2
	diff_h = (max_dim - height) // 2
	image = np.pad(image, ((diff_h, diff_h), (diff_w, diff_w)), mode='constant', constant_values=0)

	rotation_matrix = cv2.getRotationMatrix2D((max_dim // 2, max_dim // 2), angle, 1.0)
	rotated_image = cv2.warpAffine(image, rotation_matrix, (max_dim, max_dim))

	return rotated_image



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


def get_orientation_naive(section):
	section_height = section.shape[0]
	section_top = section[:section_height // 2, :]
	section_bottom = section[section_height // 2:, :]
	section_top_max = round(np.sum(section_top))
	section_bottom_max = round(np.sum(section_bottom))
	return section_top_max > section_bottom_max
