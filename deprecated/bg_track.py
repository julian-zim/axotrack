import random
import os
import numpy as np
import cv2
from utils import choose_directory, show_image


def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return (x, y)


def rotate(image, angle):
	height, width = image.shape

	max_dim = max([height, width])
	diff_h = (max_dim - height) // 2
	diff_w = (max_dim - width) // 2
	image = np.pad(image, ((diff_h, diff_h), (diff_w, diff_w)), mode='constant', constant_values=0)

	rotation_matrix = cv2.getRotationMatrix2D((max_dim // 2, max_dim // 2), angle, 1.0)
	rotated_image = cv2.warpAffine(image, rotation_matrix, (max_dim, max_dim))

	return rotated_image


def extract_background(frames, outpath):
	if os.path.exists(os.path.join(outpath, 'Background.tiff')):
		print('Found background file!')
		return

	print('Extracting background...')
	img_bg = np.median(np.array(frames).astype(np.float16), axis=0).astype(np.uint8)
	cv2.imwrite(os.path.join(outpath, 'Background.tiff'), img_bg)
	print('Done!')

def adjustment_phase(path):
	# simulate incoming camera frames
	img_names = os.listdir(path)
	num_imgs = len(img_names)
	if 'Background.tiff' in img_names:
		num_imgs -= 1
	frequency = 1
	imgs = [cv2.cvtColor(cv2.imread(os.path.join(path, img_name)), cv2.COLOR_BGR2GRAY) for img_name in img_names[::(31 - frequency)]]
	
	extract_background(imgs, path)
	

def record_phase(path, right_side, distance_x, distance_y, amplitude, speed, color):
	# simulate incoming camera frames
	img_names = os.listdir(path)  # not properly sorted fsr
	num_imgs = len(img_names)
	if 'Background.tiff' in img_names:
		num_imgs -= 1
	img_name_prefix = '_'.join(img_names[-1].split('_')[:-1])

	img_bg = cv2.cvtColor(cv2.imread(os.path.join(path, 'Background.tiff')), cv2.COLOR_BGR2GRAY)
	x, y, w, h = 0, 0 ,img_bg.shape[0], img_bg.shape[1]
	track_window = (x, y, w, h)
	term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
	
	frequency = 1
	jitter = (0., 0.)
	counter = 0
	for id in range(1, num_imgs, (31 - frequency)):
		# simulate incoming camera frames
		img = cv2.cvtColor(cv2.imread(os.path.join(path, img_name_prefix + '_' + str(id) + '.tiff')), cv2.COLOR_BGR2GRAY)
		
		show_image(img, 'img_diff_track_dot', 10)

		# track
		img_diff = cv2.subtract(img_bg, img)
		cv2.normalize(img_diff, img_diff, 0, 255, cv2.NORM_MINMAX)
		ret, track_window = cv2.CamShift(img_diff, track_window, term_crit)

		# rotation
		width, height = ret[1]
		angle = ret[2]  # angle of the width line
		fix = width < height
		if fix:	
			angle = (angle + 90) % 180
		x, y, w, h = track_window
		roi = img_diff[y:y+h, x:x+w]
		roi_rotated = rotate(roi, angle)  # rotates in the opposite direction
		#cv2.imwrite('roi.png', roi_rotated)
		roi_rotated_width = roi_rotated.shape[1]
		roi_left = roi_rotated[:, :roi_rotated_width // 2]
		roi_right = roi_rotated[:, roi_rotated_width // 2:]
		roi_left_sum = int(np.sum(roi_left))
		roi_right_sum = int(np.sum(roi_right))
		upper_hemisphere = roi_left_sum > roi_right_sum
		if upper_hemisphere:
			angle = (angle + 180) % 360
		#print(f'width: {width:.2f}, height: {height:.2f}, wa: {angle:.2f}, upper_hemisphere: {upper_hemisphere}')

		# dot
		center = np.array(ret[0])
		offset_x = np.array(pol2cart(distance_x, np.deg2rad((angle + 90) % 360)))
		offset_y = np.array(pol2cart(max([width, height]) / 2 + distance_y, np.deg2rad(angle)))
		dot = center + offset_x * (1 if right_side else -1) + offset_y
		counter += 1
		if counter >= (100 - speed):
			jitter = np.array((random.randrange(-amplitude, amplitude + 1), random.randrange(-amplitude, amplitude + 1)))
			counter = 0
		dot += jitter

		# draw
		pts = cv2.boxPoints(ret)
		img_diff_track = cv2.polylines(img, [pts.astype(np.intp)], True, (0, 0, 0) if upper_hemisphere else (255, 255, 255), 2)
		img_diff_track = cv2.line(img_diff_track, [round(c) for c in center], [round(c) for c in dot], (255, 255, 255), 1)
		img_diff_track_dot = cv2.circle(img_diff_track, [round(c) for c in dot], 1, color, 5)
		#show_image(img_diff_track_dot, 'img_diff_track_dot', 10)



def main():
	right_side = False
	distance_x = 20
	distance_y = -15
	amplitude = 5
	speed = 95
	color = (255, 255, 255)	

	path = choose_directory()
	if path == '':
		return

	adjustment_phase(path)
	record_phase(path, right_side, distance_x, distance_y, amplitude, speed, color)

		

if __name__ == '__main__':
	print('\n')
	main()
