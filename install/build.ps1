Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process -Force
conda activate axotrack
pyinstaller --name="AxoTrack" --icon="icon.ico" "../main.py" --onedir --noconsole
