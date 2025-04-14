Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process -Force
conda activate axotrack
pyinstaller --name="AxoTrack" --icon="icon.ico" "../main.py" --onedir --noconsole
Copy-Item -Path "..\manual\manual.pdf" -Destination ".\dist\AxoTrack" -ErrorAction SilentlyContinue
Start-Process "explorer.exe" -ArgumentList ".\dist\AxoTrack"

$desktopPath = [Environment]::GetFolderPath("Desktop")
Copy-Item -Path ".\dist\AxoTrack" -Destination "$desktopPath\AxoTrack\" -Recurse

$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut("$desktopPath\AxoTrack.lnk")
$shortcut.TargetPath = "$desktopPath\AxoTrack\AxoTrack.exe"
$shortcut.Save()
