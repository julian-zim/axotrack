from background_cap_sim import capture_background
from dot_projector_sim import main
import utils

if __name__ == '__main__':
	capture_background()
	utils.show_info('Press \"OK\" to start the experiment.')
	main()
