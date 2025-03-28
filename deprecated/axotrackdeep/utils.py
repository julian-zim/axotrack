import numpy
from tkinter import Tk, filedialog, messagebox
import cv2


def rescale(value, min_old, max_old, min_new, max_new):
	return (value - min_old) / (max_old - min_old) * (max_new - min_new) + min_new


def pol2cart(r, phi):
    x = r * numpy.cos(phi)
    y = r * numpy.sin(phi)
    return (x, y)


def show_image(image, window='OpenCV', duration=0):
    image = cv2.imshow('image' if not window else window, image)
    cv2.waitKey(duration)


def show_error(message=None):
	root = Tk()
	root.withdraw()
	messagebox.showerror('Error', message)


def show_info(message=None):
	root = Tk()
	root.withdraw()
	messagebox.showinfo('Information', message)
	

def choose_file(title=None):
	root = Tk()
	root.withdraw()
	file_path = filedialog.askopenfilename(title=title)
	return file_path


def choose_directory(title=None):
	root = Tk()
	root.withdraw()
	dir_path = filedialog.askdirectory(title=title)
	return dir_path


def setupCamera(camera):
	exposure_time = 20000
	binning = 2
	size = (700, 700)
	offset = (128, 50)

	camera.PixelFormat.Value = "Mono8"
	camera.AcquisitionFrameRateEnable.SetValue(True)
	camera.AcquisitionFrameRate.SetValue(30)
	camera.ExposureTime.Value = exposure_time

	camera.BinningHorizontalMode.SetValue('Sum')
	camera.BinningVerticalMode.SetValue('Sum')
	camera.BinningHorizontal.SetValue(binning)
	camera.BinningVertical.SetValue(binning)

	camera.Width.SetValue(size[0])
	camera.Height.SetValue(size[1])
	camera.CenterX.SetValue(False)
	camera.CenterY.SetValue(False)
	camera.OffsetX.SetValue(offset[0])
	camera.OffsetY.SetValue(offset[1])
