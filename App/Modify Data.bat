@echo off
python -c "print('Python is available')"
if errorlevel 1 goto :python_not_found

powershell -command "Test-Path -Path 'HKLM:\SOFTWARE\Microsoft\Windows Media Foundation\Platform' | Out-Null"
if errorlevel 1 (
    echo No webcam found.
    pause
) else (
    echo Webcam found.
)


@cd "%~dp0\..\gui codes"
@python.exe create_gui.py %*
goto :end

:python_not_found
echo Python is not installed on this system.
pause

:end
