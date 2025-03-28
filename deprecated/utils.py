from tkinter import Tk, filedialog, messagebox
import cv2


def show_image(image, title=None, duration=0):
    image = cv2.imshow('image' if not title else title, image)
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
