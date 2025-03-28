from projector_sync import main as main_ps
from background_cap import main as main_bc
from animal_track import main as main_at

def main():
	main_ps()
	main_bc()
	main_at()

if __name__ == '__main__':
	print('PROGRAM STARTED')
	main()
