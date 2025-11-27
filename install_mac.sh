#!/bin/bash

echo "=== Python-Pakete installieren (macOS) ==="

# Homebrew prüfen
if ! command -v brew &> /dev/null
then
    echo "Homebrew wird benötigt – installiere es jetzt..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# benötigte Systempakete
brew install ffmpeg
brew install libsndfile
brew install tcl-tk

# Python-Pakete
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Fertig! ==="