@echo off
setlocal

set "ROOT=%~dp0"
set "SCRIPT=%ROOT%scripts\launch_studio.py"
set "PYTHON_BIN="
set "PYTHON_LAUNCHER="

if defined ML_TEACHING_STUDIO_PYTHON (
    set "PYTHON_BIN=%ML_TEACHING_STUDIO_PYTHON%"
) else if exist "%ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%ROOT%.venv\Scripts\python.exe"
) else if exist "%ROOT%.venv\Scripts\pythonw.exe" (
    set "PYTHON_BIN=%ROOT%.venv\Scripts\pythonw.exe"
) else (
    where py >nul 2>nul && set "PYTHON_LAUNCHER=py -3"
    if not defined PYTHON_LAUNCHER (
        where python >nul 2>nul && set "PYTHON_LAUNCHER=python"
    )
)

cd /d "%ROOT%" || exit /b 1

if defined PYTHON_BIN (
    "%PYTHON_BIN%" "%SCRIPT%" %*
    set "STATUS=%ERRORLEVEL%"
) else if defined PYTHON_LAUNCHER (
    %PYTHON_LAUNCHER% "%SCRIPT%" %*
    set "STATUS=%ERRORLEVEL%"
) else (
    echo No Python interpreter was found for ML-Teaching Studio.
    pause
    exit /b 1
)

if not "%STATUS%"=="0" (
    echo.
    echo ML-Teaching Studio exited with status %STATUS%.
    pause
)

exit /b %STATUS%
