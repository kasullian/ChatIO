# ChatIO
Basic socket chat server utilizing aiohttp &amp; socketio.

# Installation
```
# install dependencies if on windows
pip install -r windows.txt

#install dependencies if on ubuntu
pip install -r ubuntu.txt

# run server, tensorflowtts will be disabled on windows
python server.py
```

# Virtual Env Installation
```
# create a virtual env for the project using your desired python version
C:\Users\username\AppData\Local\Programs\Python\Python310\python.exe -m venv venv

# activate the newly created virtual environment & install dependencies (on windows)
./venv/scripts/activate.ps1
pip install -r windows.txt

# activate the newly created virtual environment & install dependencies (on ubuntu)
./venv/scripts/activate
pip install -r ubuntu.txt

# run the server
python server.py
```
