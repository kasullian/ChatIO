# ChatIO
Basic chat service utilizing Coqui TTS, Coqui STT, aiohttp &amp; socketio.

# Requirements
* Python 3.8

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

# Virtual Environment Installation (Recommended)
```
# create a virtual env for the project using your desired python version
C:\Users\username\AppData\Local\Programs\Python\Python38\python.exe -m venv venv

# activate the newly created virtual environment & install dependencies (on windows)
pip install virtualenv
./venv/scripts/activate.ps1

# activate the newly created virtual environment & install dependencies (on ubuntu)
pip install python3.8-venv
source venv/bin/activate

pip install -r requirements.txt

# install Coqui TTS
pip install TTS

# run the server
python server.py
```

