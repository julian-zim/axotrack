import os
import numpy as np
from tkinter import Tk, filedialog, messagebox
import cv2


def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return (x, y)

def cart2pol(x, y):
	r = np.sqrt(x**2 + y**2)
	phi = np.arctan2(y, x)
	return (r, phi)

def string_to_float(string):
	# this function is implemented to handle special float string cases, like scientific notation and separating commas
	string_fixed = string.replace('.', '').replace(',', '.')
	return float(string_fixed)  # TODO: implement


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
	

def save_file(title=None, initialdir='.', filetypes=['*'], defaultextension='txt', initialfile='file.txt'):
	root = Tk() 
	root.withdraw()
	file_path = filedialog.asksaveasfilename(title=title, initialdir=initialdir, defaultextension=defaultextension, filetypes=filetypes, initialfile=initialfile)
	return file_path
	

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


def setup_camera(camera):
	#size = (1544, 1544)  # (1544, 1544)
	#offset = (0, 0)

	camera.PixelFormat.Value = "Mono8"
	camera.AcquisitionFrameRateEnable.SetValue(True)
	camera.AcquisitionFrameRate.SetValue(30)
	camera.ExposureTime.Value = 20000

	camera.BinningHorizontalMode.SetValue('Sum')
	camera.BinningVerticalMode.SetValue('Sum')
	camera.BinningHorizontal.SetValue(1)
	camera.BinningVertical.SetValue(1)

	#camera.OffsetX.SetValue(offset[0])
	#camera.OffsetY.SetValue(offset[1])
	#camera.Width.SetValue(size[0])
	#camera.Height.SetValue(size[1])
	camera.CenterX.SetValue(False)
	camera.CenterY.SetValue(False)


def extract_background():
	fps = 30
	frequency = 15
	start = 0 #- 7000
	stop = -1 #- 7000

	print('\n')
	path = choose_directory()

	print('Reading...')
	img_names = sorted(os.listdir(path), key=lambda name: int(name.split('_')[-1].split('.')[0]))

	print('Importing...')
	imgs = [cv2.cvtColor(cv2.imread(os.path.join(path, img_name)), cv2.COLOR_BGR2GRAY) for img_name in img_names[start:stop:(fps // frequency)]]
	
	print('Computing...')
	mean_img = np.mean(np.array(imgs).astype(np.float16), axis=0).astype(np.uint8)

	print('Saving...')
	os.makedirs(os.path.join(os.getcwd(), 'data'), exist_ok=True)
	suffix = f'_{path.split('/')[-1]}'
	cv2.imwrite(os.path.join(os.getcwd(), 'data', f'background{suffix}.jpg'), mean_img)
	
	print('Done!')
