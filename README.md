# ChatIO
Basic socket chat server utilizing aiohttp &amp; socketio.

# Requirements
Python 3.8

A speech to text model

# Installation
```
# install dependencies if on windows
pip install -r windows.txt

# install dependencies if on ubuntu
pip install -r ubuntu.txt

# run server, tensorflowtts will be disabled on windows
python server.py
```

# Virtual Env Installation
```
# create a virtual env for the project using your desired python version
C:\Users\username\AppData\Local\Programs\Python\Python38\python.exe -m venv venv

# activate the newly created virtual environment & install dependencies (on windows)
./venv/scripts/activate.ps1
pip install -r windows.txt

# activate the newly created virtual environment & install dependencies (on ubuntu)
./venv/scripts/activate
pip install -r ubuntu.txt

# run the server
python server.py
```

# Downloading STT Models
```
# install 🐸STT model manager
pip install -U pip
pip install coqui-stt-model-manager

# run the model manager. A browser tab will open and you can then download and test models from the Model Zoo.
stt-model-manager

# once desired model is downloaded, you'll have to either move the model & scorer into the project directory, or reference them in the code.
```
