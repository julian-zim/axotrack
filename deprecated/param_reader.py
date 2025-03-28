import tkinter as tk
from tkinter import simpledialog


class CustomDialog(simpledialog.Dialog):

	def body(self, master):
		tk.Label(master, text="Periphery:").grid(row=0)
		tk.Label(master, text="Medial Distance:").grid(row=1)
		tk.Label(master, text="Lateral Distance:").grid(row=2)
		tk.Label(master, text="Jitter Magnitude:").grid(row=3)
		tk.Label(master, text="Jitter Speed:").grid(row=4)
		tk.Label(master, text="Size:").grid(row=5)
		tk.Label(master, text="Color (RGB):").grid(row=6)

		self.param1 = tk.Entry(master)
		self.param2 = tk.Entry(master)
		self.param3 = tk.Entry(master)
		self.param4 = tk.Entry(master)
		self.param5 = tk.Entry(master)
		self.param6 = tk.Entry(master)
		self.param7 = tk.Entry(master)

		self.param1.grid(row=0, column=1)
		self.param2.grid(row=1, column=1)
		self.param3.grid(row=2, column=1)
		self.param4.grid(row=3, column=1)
		self.param5.grid(row=4, column=1)
		self.param6.grid(row=5, column=1)
		self.param7.grid(row=6, column=1)
		
		self.param1.insert(0, 'left')
		self.param2.insert(0, '20')
		self.param3.insert(0, '-15')
		self.param4.insert(0, '5')
		self.param5.insert(0, '95')
		self.param6.insert(0, '1')
		self.param7.insert(0, '255,255,255')

		return self.param1

	def apply(self):
		self.result = {
			"periphery": self.param1.get(),
			"distance_x": self.param2.get(),
			"distance_y": self.param3.get(),
			"magnitude": self.param4.get(),
			"speed": self.param5.get(),
			"size": self.param6.get(),
			"color": self.param7.get()
		}


def read_params():
	 
	root = tk.Tk()
	root.withdraw()

	dialog = CustomDialog(root, title="Dot Parameters")
	if not dialog.result:
		return None
	
	try:
		periphery = None
		if dialog.result['periphery'].lower() == 'left':
			periphery = False
		elif dialog.result['periphery'].lower() == 'right':
			periphery = True
		else:
			raise ValueError()
		distance_x = float(dialog.result['distance_x'])
		distance_y = float(dialog.result['distance_y'])
		magnitude = float(dialog.result['magnitude'])
		speed = float(dialog.result['speed'])
		size = float(dialog.result['size'])
		color = tuple([int(c) for c in dialog.result['color'].split(',')])
	except ValueError:
		print('Wrong input format!')
		return None

	root.destroy()

	return periphery, distance_x, distance_y, magnitude, speed, size, color


if __name__ == '__main__':
	params = read_params()
