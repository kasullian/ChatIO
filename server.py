import asyncio
from aiohttp import web
import socketio
import base64
import os

if os.name != 'nt': #tensorflowtts only works on ubuntu
    import soundfile as sf
    import numpy as np
    import tensorflow as tf
    from tensorflow_tts.inference import AutoProcessor
    from tensorflow_tts.inference import TFAutoModel
    processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
    tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")
    melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en")

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.event
async def chatMessage(sid, msg):
    print(f'chatMessage received from: {sid}')
    await sio.emit('chatMessage', msg)


@sio.event
async def emitText(sid, msg):    
    if os.name != 'nt': #tensorflowtts only works on ubuntu
        input_ids = processor.text_to_sequence(msg)
        #tacotron2 inference (text-to-mel)
        decoder_output, mel_outputs, stop_token_prediction, alignment_history = tacotron2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            input_lengths=tf.convert_to_tensor([len(input_ids)], tf.int32),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
        )
        #melgan inference (mel-to-wav)
        audio = melgan.inference(mel_outputs)[0, :, 0]
        #create file from melgan wav data & send it off
        sf.write('./ljs.wav', audio, 22050, "PCM_16")
    encode_string = base64.b64encode(open("ljs.wav", "rb").read()).decode()
    await sio.emit('avatarResponse', {'text': msg, 'wav': encode_string}, room=sid)


@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)
    await sio.emit('chatMessage', f'User connected: {sid}')


@sio.event
def disconnect(sid):
    print(f'Client disconnected: {sid}')


app.router.add_get('/', index)
if __name__ == '__main__':
    web.run_app(app)