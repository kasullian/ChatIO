# ChatIO
Basic chat service utilizing Coqui TTS, Google Cloud speech-to-text, GPT-3, aiohttp &amp; socketio.

# Requirements
* Python 3.8
* https://www.python.org/downloads/release/python-380/

# Configuring env variables
 Create a .env file in the project directory then paste the following into the file
```
PROJECT_ID=
GOOGLE_APPLICATION_CREDENTIALS=key.json
OPENAI_API_KEY=
```

# Virtual Environment Installation
```
# install python venv (on windows)
pip install virtualenv

# install python venv (on ubuntu)
pip install python3.8-venv

# create a virtual env for the project using your desired python version
C:\Users\username\AppData\Local\Programs\Python\Python38\python.exe -m venv venv

# activate venv windows
./venv/scripts/activate.ps1

# activate venv ubuntu
source venv/bin/activate

# install pytorch with cuda
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# install all dependencies
pip install -r requirements.txt

# install Coqui TTS
pip install TTS

# install spacy
python -m spacy download en

# run the server
python server.py
```

