import numpy as np
import cv2


def preprocessing(image, image_bg):
	image_diff = cv2.absdiff(image, image_bg)  # TODO: which way around?

	#cv2.normalize(image_diff, image_diff, 0, 255, cv2.NORM_MINMAX)
     
	_, image_bin = cv2.threshold(image_diff, 30, 255, cv2.THRESH_BINARY)
    
	image_masked = cv2.bitwise_and(image, image, mask=image_bin)
	
	return image_masked


def get_roi(image, track_window, max_track_window_size, term_crit):	
	# cam shift tracking
	rot_window, track_window = cv2.CamShift(image, track_window, term_crit)
	track_x, track_y, track_w, track_h = track_window
	(rot_cx, rot_cy), (rot_w, rot_h), angle = rot_window  # angle of the width axis w.r.t. to the x-axis

	track_w_max, track_h_max = max_track_window_size

	# limit track window size
	if track_w > track_w_max:
		track_w = track_w_max
	if track_h > track_h_max:
		track_h = track_h_max

	# fix angle & shape
	if rot_w > rot_h:  
		angle = (angle + 90) % 180
		temp = rot_w
		rot_w = rot_h
		rot_h = temp
	if rot_w > track_w_max:
		rot_w = track_w_max
	if rot_h > track_h_max:
		rot_h = track_h_max

	return ((rot_cx, rot_cy), (rot_w, rot_h), angle), (track_x, track_y, track_w, track_h)


def roi_backrotation(image, angle):
	height, width = image.shape

	max_dim = max([width, height])
	diff_w = (max_dim - width) // 2
	diff_h = (max_dim - height) // 2
	image = np.pad(image, ((diff_h, diff_h), (diff_w, diff_w)), mode='constant', constant_values=0)

	rotation_matrix = cv2.getRotationMatrix2D((max_dim // 2, max_dim // 2), angle, 1.0)
	rotated_image = cv2.warpAffine(image, rotation_matrix, (max_dim, max_dim))

	return rotated_image


def get_orientation_naive(roi):
	section_height = roi.shape[0]
	section_top = roi[:section_height // 2, :]
	section_bottom = roi[section_height // 2:, :]
	section_top_max = round(np.sum(section_top))
	section_bottom_max = round(np.sum(section_bottom))
	return section_top_max > section_bottom_max


def get_orientation(roi):
	# TODO
	angle = 0
	center = 0
	return angle, center
