import asyncio

from aiohttp import web

import socketio
import base64

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
async def wavMessage(sid, msg):
    #todo generate tts wav based on msg
    encode_string = base64.b64encode(open("ljs.wav", "rb").read()).decode()
    await sio.emit('wavMessage', encode_string, room=sid)


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