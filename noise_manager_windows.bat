@echo off
echo Starte Linux-Installer in WSL...
wsl bash install_linux.sh
pause
python .\noise_manager.py
pause