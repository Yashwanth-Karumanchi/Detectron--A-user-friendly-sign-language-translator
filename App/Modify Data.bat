@echo off
python -c "print('Python is available')"
if errorlevel 1 goto :python_not_found

@cd /d "%~dp0\..\gui codes"
@python.exe create_gui.py %*
goto :end

:python_not_found
echo Python is not installed on this system.
pause

:end
