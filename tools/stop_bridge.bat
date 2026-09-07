@echo off
python "%~dp0antigravity_bridge.py" --disable
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *antigravity_bridge*" 2>nul
echo Antigravity Bridge stopped and proxy settings restored to direct.
pause
