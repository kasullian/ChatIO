import requests
from asyncio.windows_events import NULL
from aiohttp import web
import socketio
import base64
import json
from websocket import create_connection
import wave
from timeit import default_timer as timer
import numpy as np
from stt import Model
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import os
from dotenv import load_dotenv
load_dotenv()

# get TTS models from coqui / https://github.com/coqui-ai/TTS/blob/main/TTS/bin/synthesize.py
response = requests.get('https://raw.githubusercontent.com/coqui-ai/TTS/main/TTS/.models.json')
data = response.content.decode('utf-8', errors="replace")
with open('models.json', 'w', encoding='utf-8') as f:
    f.write(data)
manager = ModelManager('models.json')
#manager.list_models()
model_path, config_path, model_item = manager.download_model('tts_models/en/vctk/vits')
speakers_file_path = None
language_ids_file_path = None
vocoder_path = None
vocoder_config_path = None
encoder_path = None
encoder_config_path = None

# load TTS model
synthesizer_load_start = timer()
synthesizer = Synthesizer(
    model_path,
    config_path,
    speakers_file_path,
    language_ids_file_path,
    vocoder_path,
    vocoder_config_path,
    encoder_path,
    encoder_config_path,
    False, #cuda
)
synthesizer_load_end = timer() - synthesizer_load_start
print("Loaded synthesizer in {:.3}s.".format(synthesizer_load_end))

# load speech to text model
scorer = "coqui-stt-0.9.3-models.scorer"
model = "model.tflite"
model_load_start = timer()
ds = Model(model)
model_load_end = timer() - model_load_start
print("Loaded model in {:.3}s.".format(model_load_end))

# load speech to text scorer
scorer_load_start = timer()
ds.enableExternalScorer(scorer)
scorer_load_end = timer() - scorer_load_start
print("Loaded scorer in {:.3}s.".format(scorer_load_end))

# initialize blenderbot instance
try:    
    ws = create_connection(os.getenv('BLENDERBOT_URL')) #dirty, should close at some point ######
    ws.send(json.dumps({"text": "begin"})) #sequence to initiate basic parl-ai socket instance###
    result =  ws.recv()                    #todo: replace with client class for instancing ######
    ws.send(json.dumps({"text": "begin"})) ######################################################
    result = ws.recv()                     ######################################################
    print("Received '%s'" % result)        ######################################################
except:
    ws = 0
    print("Unable to connect to BlenderBot.")

# initialize socketio web server
sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

@sio.event
async def chatMessage(sid, msg):
    print(f'chatMessage received from: {sid}')
    if ws != 0:
        await sio.emit('chatMessage', {"User": msg})
        ws.send(json.dumps({"text": f"{msg}"}))
        result = ws.recv()
        print("Received '%s'" % result)
        bbResponse = json.loads(result)
        await sio.emit('chatMessage', {"BlenderBot": bbResponse["text"]})
    else:
        await sio.emit('chatMessage', 'Not connected to BlenderBot.') 

@sio.event
async def emitText(sid, msg):    
    ws.send(json.dumps({"text": f"{msg}"}))
    result = ws.recv()
    bbResponse = json.loads(result)
    wav = synthesizer.tts(bbResponse["text"], "p243")
    synthesizer.save_wav(ttswav, f"{sid}.wav")
    encode_string = base64.b64encode(open(f"{sid}.wav", "rb").read()).decode()
    os.remove(f"{sid}.wav")
    await sio.emit('avatarResponse', {'text': msg, 'wav': encode_string}, room=sid)

@sio.event #todo: stop writing wav to disk
async def emitAudio(sid, msg):
    decoded_data = base64.b64decode(msg)
    with wave.open(f"{sid}.wav", 'w') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
        wav.writeframes(decoded_data)
    with wave.open(f"{sid}.wav", 'r') as wav:
        fs_orig = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), np.int16)
        audio_length = wav.getnframes() * (1 / fs_orig)
        inference_start = timer()
        out = ds.stt(audio)
        inference_end = timer() - inference_start
        print("STT Inference took %0.3fs for %0.3fs audio file." % (inference_end, audio_length))
        print(f"User said: {out}")
        blender_start = timer()
        out = ds.stt(audio)
        ws.send(json.dumps({"text": f"{out}"}))
        result = ws.recv()
        blender_end = timer() - blender_start
        print("BB Inference took %0.3fs" % (blender_end))
        bbResponse = json.loads(result)["text"]
        ttswav = synthesizer.tts(bbResponse, "p243")
        synthesizer.save_wav(ttswav, f"{sid}.wav")
    encode_string = base64.b64encode(open(f"{sid}.wav", "rb").read()).decode()
    os.remove(f"{sid}.wav")
    await sio.emit('avatarResponse', {'text': msg, 'wav': encode_string}, room=sid)

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)
    await sio.emit('chatMessage', f'User connected: {sid}')

@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')

# serve html
async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

app.router.add_get('/', index)
if __name__ == '__main__':
    web.run_app(app, port=1337)