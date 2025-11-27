echo "=== Python-Pakete installieren (Linux) ==="

# benötigte Systempakete
sudo apt update
sudo apt install -y ffmpeg libsndfile1 python3-tk

# Python-Pakete
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Fertig! ==="