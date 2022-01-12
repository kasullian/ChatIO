import requests
from asyncio.windows_events import NULL
from aiohttp import web
import socketio
import base64
import os
import json
from websocket import create_connection
import wave
from timeit import default_timer as timer
import numpy as np
from stt import Model

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

# load speech to text model
scorer = "coqui-stt-0.9.3-models.scorer"
model = "model.tflite"
print("Loading model from file {}".format(model))
model_load_start = timer()
ds = Model(model)
model_load_end = timer() - model_load_start
print("Loaded model in {:.3}s.".format(model_load_end))

# load speech to text scorer
print("Loading scorer from files {}".format(scorer))
scorer_load_start = timer()
ds.enableExternalScorer(scorer)
scorer_load_end = timer() - scorer_load_start
print("Loaded scorer in {:.3}s.".format(scorer_load_end))

# serve html
async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

ws = NULL
try:    
    ws = create_connection("ws://localhost:8080/websocket") #dirty, should close at some point
    ws.send(json.dumps({"text": "begin"})) #sequence to initiate basic parl-ai socket instance
    result =  ws.recv()                    ##################################################
    ws.send(json.dumps({"text": "begin"})) ##################################################
    result = ws.recv()                     ##################################################
    print("Received '%s'" % result)        ##################################################
except:
    print("Unable to connect to BlenderBot.")

@sio.event
async def chatMessage(sid, msg):
    print(f'chatMessage received from: {sid}')
    await sio.emit('chatMessage', {"User": msg})
    ws.send(json.dumps({"text": f"{msg}"}))
    result = ws.recv()
    print("Received '%s'" % result)
    bbResponse = json.loads(result)
    await sio.emit('chatMessage', {"BlenderBot": bbResponse["text"]})


@sio.event
async def emitText(sid, msg):    
    ws.send(json.dumps({"text": f"{msg}"}))
    result = ws.recv()
    print("Received '%s'" % result)
    bbResponse = json.loads(result)
    params = {
        'text': {bbResponse["text"]},
        'speaker_id': 'p243',
        'style_wav': ''
    }
    r = requests.get("http://localhost:5002/api/tts", params=params)
    open('ljs.wav', 'wb').write(r.content)
    encode_string = base64.b64encode(open("ljs.wav", "rb").read()).decode()
    await sio.emit('avatarResponse', {'text': msg, 'wav': encode_string}, room=sid)


@sio.event
async def emitAudio(sid, msg):
    decoded_data = base64.b64decode(msg)
    with wave.open("out.wav", 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wav.writeframes(decoded_data)
        fs_orig = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
        audio_length = wav.getnframes() * (1 / fs_orig)
        inference_start = timer()
        out = ds.stt(audio)
        inference_end = timer() - inference_start
        print("Inference took %0.3fs for %0.3fs audio file." % (inference_end, audio_length))
        await sio.emit('chatMessage', {"User": out})


@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)
    await sio.emit('chatMessage', f'User connected: {sid}')


@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')


app.router.add_get('/', index)
if __name__ == '__main__':
    web.run_app(app, port=1337)