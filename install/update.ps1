Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process -Force

git pull https://github.com/julian-zim/axotrack.git

if (Test-Path ".\build\AxoTrack") {
    Remove-Item ".\build\AxoTrack" -Recurse -Force
}
if (Test-Path ".\dist\AxoTrack") {
    Remove-Item ".\dist\AxoTrack" -Recurse -Force
}
if (Test-Path ".\AxoTrack.spec") {
    Remove-Item ".\AxoTrack.spec" -Force
}
pyinstaller --name="AxoTrack" --icon="icon.ico" "../main.py" --onedir --noconsole
Copy-Item -Path "..\manual\manual.pdf" -Destination ".\dist\AxoTrack" -ErrorAction SilentlyContinue

$desktopPath = [Environment]::GetFolderPath("Desktop")

$folderPath = "$desktopPath\AxoTrack"
if (Test-Path $folderPath) {
    Remove-Item $folderPath -Recurse -Force
}
Copy-Item -Path ".\dist\AxoTrack" -Destination $folderPath -Recurse

$shortcutPath = "$desktopPath\AxoTrack.lnk"
if (Test-Path $shortcutPath) {
    Remove-Item $shortcutPath -Force
}
$wshShell = New-Object -ComObject WScript.Shell
$shortcut = $wshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$folderPath\AxoTrack.exe"
$shortcut.Save()

Start-Process "explorer.exe" -ArgumentList $folderPath
