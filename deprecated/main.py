from background_cap_home import capture_background
from projector_sync import sync_projector
from param_reader import read_params
from track_loop_home import track
from dot_projector import project_dot


if __name__ == '__main__':
	capture_background()
	sync_projector()
	result = read_params()
	if result is None:
		raise ValueError()
	periphery, distance_x, distance_y, magnitude, speed, size, color = result
	track_window = track(None, None, None, None, None, None, None)
	project_dot()
