import numpy as np
from screeninfo import get_monitors
import cv2
import utils
#from pypylon import pylon


def main():
	image_size = (1000, 1000)  # (1544, 1544)

	monitors = get_monitors()
	if len(monitors) < 2:
		utils.show_error('No second screen detected!')
		return
	
	projwindow_name = 'Projection Window'
	cv2.namedWindow(projwindow_name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(projwindow_name, monitors[1].x, monitors[1].y)
	cv2.setWindowProperty(projwindow_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	pad_width = round((image_size[1] * monitors[1].width / monitors[1].height - image_size[0]) / 2)

	trackwindow_name = 'Control Window'
	cv2.namedWindow(trackwindow_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(trackwindow_name, round(monitors[1].width * 0.75), round(monitors[1].height * 0.75))
	cv2.moveWindow(trackwindow_name, monitors[0].x + monitors[0].height * 1 // 10, monitors[0].y + monitors[0].height * 1 // 10)

	#camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
	#camera.Open()
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():  #camera.IsOpen()
		utils.show_error('Cannot find camera.')
		return
	#utils.setupCamera(camera)

	# TODO: add zoom by camera cropping & offset
	#camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
	while cap.isOpened():  #camera.IsGrabbing()
		#grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
		#if not grabResult.GrabSucceeded(): break
		#frame = cv2.flip(grabResult.Array, -1)
		frame =  cv2.flip(cap.read()[1], -1)
		#grabResult.Release()
		if len(frame.shape) != 2:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		image = cv2.resize(frame, image_size)
		image = np.pad(image, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)

		cv2.imshow(trackwindow_name, image)
		cv2.imshow(projwindow_name, image)
		cv2.waitKey(1)


main()
