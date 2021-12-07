import asyncio

from aiohttp import web

import socketio

sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.event
async def chatMessage(sid, message):
    print('chatMessage received from:', sid)

    #emit users message
    await sio.emit('chatMessage', message, room=sid)
    
    #pass message through blenderbot
    #todo

    #emit blenderbot response
    #todo
    #await sio.emit('chatMessage', {'data': message, 'debug': True, 'sid': sid}, room=sid)


@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)
    await sio.emit('chatMessage', f'User connected: {sid}', room=sid)


@sio.event
def disconnect(sid):
    print('Client disconnected:', sid)


app.router.add_get('/', index)
if __name__ == '__main__':
    web.run_app(app)