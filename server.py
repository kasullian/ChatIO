import os
import io
import os.path
import requests
import socketio
import asyncio
import aiohttp_cors
import base64
import json
import wave
import numpy as np
from os import path
#from stt import Model
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from aiohttp import web
from timeit import default_timer as timer
from websocket import create_connection
from tinydb import TinyDB
from dotenv import load_dotenv
from google.cloud import speech
load_dotenv()

#import torch
#print(torch.cuda.is_available())
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))

google_stt_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    enable_automatic_punctuation=True,
    audio_channel_count=1,
    language_code="en-US"
)

# google speech to text
def speech_to_text(config, audio):
    client = speech.SpeechClient()
    response = client.recognize(config=config, audio=audio)
    if response.results:
        for result in response.results:
            best_alternative = result.alternatives[0]
            transcript = best_alternative.transcript
            confidence = best_alternative.confidence
            print("-" * 80)
            print(f"STT Transcript: {transcript}")
            print(f"STT Confidence: {confidence:.0%}")
            return transcript
    else:
        return False

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
print("Loaded TTS synthesizer in {:.3}s.".format(synthesizer_load_end))

# initialize socketio web server
sio = socketio.AsyncServer(async_mode='aiohttp')
app = web.Application()
sio.attach(app)

# database
db = TinyDB('db.json')

# blenderbot socket instances
clientTable = []
def get_client(sid):
    for client in clientTable:
        if client.get_sid() == sid:
            return client
    return False

class BlenderBotClient:
    # constructor
    def __init__(self, sid):
        self.sid = sid
        self.responses = []
        self.messages = []
        self.context = '' # update context with stored data 
    def get_sid(self):
        return self.sid
    def get_logs(self):
        return json.dumps({'messages': self.messages, 'responses': self.responses})

async def select_chat(request):
    data = await request.json()
    table = db.table('logs')
    logs = table.all()
    return web.Response(text=json.dumps({'id': data.get("id"), 'logs': logs[data.get("id")]}), content_type="application/json")

async def get_chats(request):
    table = db.table('logs')
    logs = table.all()
    return web.Response(text=json.dumps(logs), content_type="application/json")

# `aiohttp_cors.setup` returns `aiohttp_cors.CorsConfig` instance.
# The `cors` instance will store CORS configuration for the
# application.
cors = aiohttp_cors.setup(app)

# To enable CORS processing for specific route you need to add
# that route to the CORS configuration object and specify its
# CORS options.
resource = cors.add(app.router.add_resource("/select-chat"))
route = cors.add(
    resource.add_route("POST", select_chat), {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers=("*"),
            allow_headers=("*")
        )
    })

resource = cors.add(app.router.add_resource("/get-chats"))
route = cors.add(
    resource.add_route("GET", get_chats), {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers=("*"),
            allow_headers=("*")
        )
    })

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from keybert import KeyBERT

model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v1')
classifier = pipeline("zero-shot-classification", model='valhalla/distilbart-mnli-12-3')
kw_model = KeyBERT()

# %%
def zeroshot_topic(sequence):
    
    labels = [
        'work', 'family', 'relationships', 'people', 'places', 'foods', 'interests', 
        # 'watch list', 'music', 'dislikes',
    ]

    results = classifier(sequence, labels, multi_label=True)
   
    label = results['labels'][0]

    output = {'sequence': sequence, 'topic': label}
    return output['topic']

def keybert_keyword(sequence, max_ngram=3):
    n = 1
    candidate_keys = []
    candidate_scores = []
    while n <= max_ngram:
        top_key = kw_model.extract_keywords(sequence, keyphrase_ngram_range=(1, n), stop_words=None)[0]
        candidate_keys += [top_key[0]]
        candidate_scores += [top_key[1]]
        n += 1

    best_score = max(candidate_scores)
    best_keyword = candidate_keys[candidate_scores.index(best_score)]

    return best_keyword

def substring_after(s, delim):
    return s.partition(delim)[2]

context="The following is a conversation with an AI assistant named Becky. The assistant is helpful, creative, clever, and very friendly.\nInformation about the Human:\nInformation about the AI:\n"
def duplicate_detection(sequence, outArray, lineNum, threshold=0.7):
    
    # Get topic and keyword from user input
    # topic = zeroshot_topic(sequence)
    new_kw = keybert_keyword(sequence, max_ngram=12)
    
    aiContext = substring_after(context, "Information about the AI:\n")
    aiContextArr = aiContext.splitlines()
    print(aiContextArr)

    # Get related keywords from user database
    # replace user_data with global context history array
    #related_data = user_data[user_data.topic == topic]
    old_keyword = aiContextArr
    #
    ## Get the embeddings and calculate similarity
    if len(aiContextArr) != 0:
        embeddings1 = model.encode(new_kw, convert_to_tensor=True)
        embeddings2 = model.encode(old_keyword, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        if max(cosine_scores[0]) < threshold:
            outArray.insert(lineNum + 1, new_kw)
            decision = "Get new keyword"
        else:
            decision = "Duplicate detected"
    else:
        outArray.insert(lineNum + 1, new_kw)
        decision = "Get new keyword"
    return decision, sequence

import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner','textcat']) #python -m spacy download en
def postagger_type(sequence):
    doc = nlp(sequence)
    noun_count = 0
    for token in doc:
        if token.pos_=='NOUN' or token.pos_=='PROPN':
            noun_count += 1
    if noun_count == 0:
        return 'Simple'
    else:
        return 'Complex'

def preprocess_input(msg):
    # todo: store user contextual keywords
    complexity = postagger_type(msg)
    if(complexity == 'Complex'):
        # topic = zeroshot_topic(msg)
        # keyword = keybert_keyword(msg)
        # print(topic, keyword)
        lineNum = 0
        global context
        for line in context.splitlines():
            print(lineNum, line)
            lineNum = lineNum + 1

    return msg

def postprocess_output(msg):
    # determine topic label
    newMsg = msg.replace('\n', '')
    complexity = postagger_type(newMsg) # if output is complex then store it in context
    global context
    if(complexity == 'Complex'):
        linesArray = context.splitlines()
        lineNum = 0
        for line in linesArray:
            if line == "Information about the AI:":
                print(duplicate_detection(newMsg, linesArray, lineNum))
                context = "\n".join(linesArray)
            lineNum = lineNum + 1
    return newMsg

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
def generate_response(msg):
    start_sequence = "\nAI: "
    restart_sequence = "\nHuman: "
    message = preprocess_input(msg)
    chat = context + restart_sequence + message + start_sequence
    response = openai.Completion.create(
      engine="text-davinci-001",
      prompt=chat,
      temperature=0.9,
      max_tokens=95,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      stop=[" Human:", " AI:"]
    )
    jsonData = json.loads(json.dumps(response))
    response = postprocess_output(jsonData.get("choices")[0].get("text"))
    return response

@sio.event
async def chatMessage(sid, msg):
    client = get_client(sid)
    if client:
        await sio.emit('chatMessage', {"User": msg}, room=sid)
        response = generate_response(msg)
        await sio.emit('chatMessage', {"Bot": response}, room=sid)
    else:
        await sio.emit('chatMessage', 'Not connected to BlenderBot.', room=sid) 

@sio.event
async def emitText(sid, msg):  
    client = get_client(sid)
    if client:
        jsonData = json.loads(json.dumps(msg))
        textIn = jsonData.get("text_input")
        response = generate_response(textIn)
        ttswav = synthesizer.tts(response, "p243")
        synthesizer.save_wav(ttswav, f"{sid}.wav")
        encode_string = base64.b64encode(open(f"{sid}.wav", "rb").read()).decode()
        os.remove(f"{sid}.wav")
        await sio.emit('avatarResponse', {'text': response, 'wav': encode_string, 'id': jsonData.get("id"), 'textInput': textIn}, room=sid)

@sio.event #todo: stop writing wav to disk
async def emitAudio(sid, msg):
    jsonData = json.loads(json.dumps(msg))
    decoded_data = base64.b64decode(jsonData.get("audio_input"))
    client = get_client(sid)
    if client:
        with wave.open(f"{sid}.wav", 'w') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
            wav.writeframes(decoded_data)
        with io.open(f"{sid}.wav", "rb") as audio_file:
            content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            inference_start = timer()
            inference_end = timer() - inference_start
            sttResponse = speech_to_text(google_stt_config, audio)
            inference_end = timer() - inference_start
            # Reads the response
            if sttResponse:
                print("STT Inference took: %0.3fs" % (inference_end))
                blender_start = timer()
                response = generate_response(sttResponse)
                blender_end = timer() - blender_start
                print("BB Inference took %0.3fs" % (blender_end))
                ttswav = synthesizer.tts(response, "p243")
                synthesizer.save_wav(ttswav, f"{sid}.wav")
                encode_string = base64.b64encode(open(f"{sid}.wav", "rb").read()).decode()
                os.remove(f"{sid}.wav")
                await sio.emit('avatarResponse', {'text': response, 'wav': encode_string, 'id': jsonData.get("id"), 'textInput': f"{sttResponse}"}, room=sid)
            else:
                ttswav = synthesizer.tts("Sorry, could you say that again?", "p243")
                synthesizer.save_wav(ttswav, f"{sid}.wav")
                encode_string = base64.b64encode(open(f"{sid}.wav", "rb").read()).decode()
                os.remove(f"{sid}.wav")
                await sio.emit('avatarResponse', {'text': 'Sorry, could you say that again?', 'wav': encode_string, 'id': jsonData.get("id"), 'textInput': 'NULL'}, room=sid)
    else:
        ttswav = synthesizer.tts("BlenderBot is offline.", "p243")
        synthesizer.save_wav(ttswav, f"{sid}.wav")
        encode_string = base64.b64encode(open(f"{sid}.wav", "rb").read()).decode()
        os.remove(f"{sid}.wav")
        await sio.emit('avatarResponse', {'text': 'BlenderBot is offline.', 'wav': encode_string, 'id': jsonData.get("id"), 'textInput': 'NULL'}, room=sid)

@sio.event
async def emitRating(sid, msg):
    jsonData = json.loads(json.dumps(msg))
    rating = jsonData.get('rating') # log rating for conversation
    text = jsonData.get('text')
    await sio.emit('chatMessage', f'User {sid} sent rating {rating} for: {text}', room=sid)

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)
    clientTable.append(BlenderBotClient(sid))
    await sio.emit('chatMessage', f'User connected: {sid}', room=sid)

@sio.event
async def disconnect(sid):
    client = get_client(sid)
    if client:
        client.disconnect()
        logs = json.loads(client.get_logs())
        if len(logs.get("responses")):
            table = db.table('logs')
            table.insert(logs)
        clientTable.remove(client)
        print(f'Client disconnected: {sid}')

# serve html
async def index(request):
    with open('app.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

app.router.add_get('/', index)
if __name__ == '__main__':
    web.run_app(app, port=1337)