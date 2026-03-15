#!/bin/bash
echo "==========================================="
echo " Booting DeSIDE-DDI AI Engine (Mac/Linux)"
echo "==========================================="

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Setting it up now..."
    # TF requires 3.11 or lower
    python3.11 -m venv venv
    source venv/bin/activate
    echo "Installing Dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "==========================================="
echo " Starting Flask Server on Port 5002..."
echo " Access the App at: http://127.0.0.1:5002"
echo "==========================================="

# Open browser if possible
if command -v open > /dev/null; then
    sleep 3 && open http://127.0.0.1:5002 &
elif command -v xdg-open > /dev/null; then
    sleep 3 && xdg-open http://127.0.0.1:5002 &
fi

python app.py
