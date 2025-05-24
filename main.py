from projector_sync import main as main_ps
from background_cap import main as main_bc
from animal_track import main as main_at

def main():
	ret_ps = main_ps()
	if ret_ps < 0:
		return
	ret_bc = main_bc()
	if ret_bc < 0:
		return
	ret_at = main_at()
	if ret_at < 0:
		return

if __name__ == '__main__':
	print('PROGRAM STARTED')
	main()
