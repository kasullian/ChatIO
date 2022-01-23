# ChatIO
Basic chat service utilizing Coqui TTS, Coqui STT, aiohttp &amp; socketio.

# Requirements
* Python 3.8
* A speech to text model

# Configuring Speech To Text
You can use a pre-build STT model from the [Coqui Model Zoo](https://coqui.ai/models/).

* Modify the source to use the url of your desired model & scorer

# Configuring BlenderBot
 Create a .env file in the project directory then paste the following into the file
```
BLENDERBOT_URL=ws://localhost:8080/websocket
```

# Installation
```
# install dependencies if on windows
pip install -r requirements.txt

# install Coqui TTS
pip install TTS

# run the server
python server.py
```

# Virtual Environment Installation
```
# create a virtual env for the project using your desired python version
C:\Users\username\AppData\Local\Programs\Python\Python38\python.exe -m venv venv

# activate the newly created virtual environment & install dependencies (on windows)
./venv/scripts/activate.ps1
pip install -r windows.txt

# activate the newly created virtual environment & install dependencies (on ubuntu)
./venv/scripts/activate
pip install -r ubuntu.txt

# install Coqui TTS
pip install TTS

# run the server
python server.py
```

