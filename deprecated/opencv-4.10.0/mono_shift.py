import time
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def convert_temp(image):
	image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	_, _, image_v = cv.split(image_hsv)
	image_v_inv = 255 - image_v # type: ignore
	return image_v_inv


def convert(image):
	image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
	image_v_con = cv.inRange(image_hsv, np.array([0., 0., 100.]), np.array([180., 255., 255.]))
	image_v_con_inv = 255 - image_v_con # type: ignore
	return image_v_con_inv


cap = cv.VideoCapture(0)
ret, frame = cap.read()

height, width, _ = frame.shape
x, y, w, h = round(width / 2) - 200, round(height / 2) - 200, 400, 300
track_window = (x, y, w, h)

term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

ret, frame = cap.read()
while ret:
	frame_conv = convert(frame)

	ret, track_window = cv.CamShift(frame_conv, track_window, term_crit)
	# Draw it on image
	pts = cv.boxPoints(ret)
	pts = pts.astype(np.intp)
	frame_track = cv.polylines(frame_conv, [pts], True, (255, 255, 255), 2)
	cv.imshow('frame_track', frame_track)
	cv.waitKey(1)

	ret, frame = cap.read()


cap.release()
