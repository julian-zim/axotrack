from background_cap import capture_background
from projector_sync import sync_projector
from dot_projector import main
import utils

if __name__ == '__main__':
	capture_background()
	sync_projector()
	utils.show_info('Press \"OK\" to start the experiment.')
	main()
