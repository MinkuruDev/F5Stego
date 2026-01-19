# F5Stego

F5Stego is a Python implementation of the F5 steganography algorithm for hiding and extracting messages inside JPEG images. It provides a simple command-line interface to embed plaintext into cover images and to extract them later.

## Features

- Embed text into JPEG images using the F5 algorithm.
- Extract hidden text from stego JPEGs.
- Command-line interface. 
- Configurable password for the embedding process.

## Requirements

- Python 3.13.11 or newer
- Additional lib in requirements.txt

## Installation

Clone the repository and install dependencies in a virtual environment (recommended):

```bash
git clone https://github.com/MinkuruDev/F5Stego.git
cd F5Stego
```

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows Command Prompt (CMD):
```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Linux / macOS (bash / zsh):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

- Embed a message:
```bash
python f5.py e -i <cover.jpg> -m <message> -o <stego.jpg> -p <password>
```

- Extract a message:
```bash
python f5.py x -i <stego.jpg> -p <password> 
```

**Examples:** 

```bash
# Embed plain text message into ./img/UAE23.png producing ./img/UAE23_stego.jpg
# with the password "Viet Nam" and keeping the temp jpg file for compute different
python f5.py embed --input ./img/UAE23.png --output ./img/UAE23_stego.jpg --message "Doi tuyen U23 Viet Nam gianh chien thang 3-2 truoc doi tuyen U23 UAE" --password "Viet Nam" --keep

# Extract:
python f5.py extract --input ./img/UAE23_stego.jpg --password "Viet Nam"

# Compute different:
python diff.py ./img/UAE23_stego.jpg ./temp/UAE23_temp.jpg --output ./temp/diff.png

# Compute different then Maximize it for better visualization
python diff.py ./img/UAE23_stego.jpg ./temp/UAE23_temp.jpg --output ./temp/diff_max.png --maximize
```

## Guess the key

Can you get the embed message in `./img/Almond_Eye_stego.png`

Hints:
1. A single word in English
2. First character is in Upper case
3. 6
4. A Fruit
