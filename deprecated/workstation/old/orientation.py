import numpy as np


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
