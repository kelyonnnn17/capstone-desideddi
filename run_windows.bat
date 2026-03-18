@echo off
echo ===========================================
echo  Booting DeSIDE-DDI AI Engine (Windows)
echo ===========================================

IF NOT EXIST venv (
    echo Virtual environment not found. Setting it up now...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing Dependencies...
    pip install -r requirements.txt
) ELSE (
    call venv\Scripts\activate.bat
)

echo ===========================================
echo  Starting Flask Server on Port 5003...
echo  Access the App at: http://127.0.0.1:5003
echo ===========================================

:: Start background browser call
start http://127.0.0.1:5003

python app.py
pause
