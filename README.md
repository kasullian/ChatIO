# ChatIO
Basic chat service utilizing Coqui TTS, Google Cloud speech-to-text, GPT-3, aiohttp &amp; socketio.

# Requirements
* Python 3.8

# Configuring OpenAI's GPT
 Create a .env file in the project directory then paste the following into the file
```
OPENAI_API_KEY=youropenaiapikey
```

# Virtual Environment Installation
```
# create a virtual env for the project using your desired python version
C:\Users\username\AppData\Local\Programs\Python\Python38\python.exe -m venv venv

# activate the newly created virtual environment & install dependencies (on windows)
pip install virtualenv
./venv/scripts/activate.ps1

# activate the newly created virtual environment & install dependencies (on ubuntu)
pip install python3.8-venv
source venv/bin/activate

# install all dependencies
pip install -r requirements.txt

# install Coqui TTS
pip install TTS

# install spacey
python -m spacy download en

# run the server
python server.py
```

