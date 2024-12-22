#!/usr/bin/env python
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from models import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(ref_dicts):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=25)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref = model.style_encoder(mel_tensor.unsqueeze(1))
        reference_embeddings[key] = (ref.squeeze(1), audio)
    
    return reference_embeddings


# Load models
import sys
sys.path.insert(0, "./Demo/hifi-gan")

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from attrdict import AttrDict
from vocoder import Generator
import librosa
import numpy as np
import torchaudio

h = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

cp_g = scan_checkpoint("Vocoder/LibriTTS/", 'g_')

config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
with open(config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)

device = torch.device(device)
generator = Generator(h).to(device)

state_dict_g = load_checkpoint(cp_g, device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()


# load StyleTTS
model_path = "./Models/VCTK/epoch_2nd_00100.pth"
model_config_path = "./Models/VCTK/config.yml"

config = yaml.safe_load(open(model_config_path))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

params = torch.load(model_path, map_location='cpu')
params = params['net']
for key in model:
    if key in params:
        if not "discriminator" in key:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key])
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]


### Conversion (seen speakers)
# get first 3 validation sample as references
train_path = config.get('train_data', None)
val_path = config.get('val_data', None)
train_list, val_list = get_data_path_list(train_path, val_path)

ref_dicts = {}
for j in range(3):
    filename = val_list[j].split('|')[0]
    name = filename.split('/')[-1].replace('.wav', '')
    ref_dicts[name] = filename
    
reference_embeddings = compute_style(ref_dicts)

# get last validation sample as input 
filename = val_list[-1].split('|')[0]
audio, source_sr = librosa.load(filename, sr=24000)
audio, index = librosa.effects.trim(audio, top_db=25)
audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32
source = preprocess(audio).to(device)

converted_samples = {}

with torch.no_grad():
    mel_input_length = torch.LongTensor([source.shape[-1]])
    asr = model.mel_encoder(source)
    F0_real, _, F0 = model.pitch_extractor(source.unsqueeze(1))
    real_norm = log_norm(source.unsqueeze(1)).squeeze(1)
    
    for key, (ref, _) in reference_embeddings.items():
        out = model.decoder(asr, F0_real.unsqueeze(0), real_norm, ref.squeeze(1))

        c = out.squeeze()
        y_g_hat = generator(c.unsqueeze(0))
        y_out = y_g_hat.squeeze()

        converted_samples[key] = y_out.cpu().numpy()

print(converted_samples)
### Conversion (unseen speakers)
# get first 3 test sample as references

test_path = val_path.replace('/val_list.txt', '/test_list.txt')
_, test_list = get_data_path_list(train_path, test_path)

ref_dicts = {}
for j in range(3):
    filename = test_list[j].split('|')[0]
    name = filename.split('/')[-1].replace('.wav', '')
    ref_dicts[name] = filename
    
reference_embeddings = compute_style(ref_dicts)

# get last test sample as input 
filename = test_list[-1].split('|')[0]
audio, source_sr = librosa.load(filename, sr=24000)
audio, index = librosa.effects.trim(audio, top_db=30)
audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32
source = preprocess(audio).to(device)


converted_samples = {}

with torch.no_grad():
    mel_input_length = torch.LongTensor([source.shape[-1]])
    asr = model.mel_encoder(source)
    F0_real, _, F0 = model.pitch_extractor(source.unsqueeze(1))
    real_norm = log_norm(source.unsqueeze(1)).squeeze(1)
    
    for key, (ref, _) in reference_embeddings.items():
        out = model.decoder(asr, F0_real.unsqueeze(0), real_norm, ref.squeeze(1))

        c = out.squeeze()
        y_g_hat = generator(c.unsqueeze(0))
        y_out = y_g_hat.squeeze()

        converted_samples[key] = y_out.cpu().numpy()


print(converted_samples)

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import gtts
import uuid
app = FastAPI()

TTS_DIR = "result/tts"
os.makedirs(TTS_DIR, exist_ok=True)

app = FastAPI()

app.mount("/result", StaticFiles(directory="result"), name="result")

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>StyleTTS-VC</title>
    </head>
    <body>
        <h1>StyleTTS-VC</h1>
        <form action="/inference" method="post" enctype="multipart/form-data">
            <label for="target_video">Target Video</label>
            <input type="file" id="target_video" name="target_video" accept="video/*" required><br><br>

            <label for="source_audio">Source Audio</label>
            <input type="file" id="source_audio" name="source_audio" accept="audio/*" required><br><br>

            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/tts")
async def tts(request: dict):
    text = request.get('text')
    if not text:
        return JSONResponse(status_code=400, content={'error': 'text field is required'})
    
    filename = f"{TTS_DIR}/{uuid.uuid4()}.mp3"
    tts = gtts.gTTS(text)
    tts.save(filename)
    return JSONResponse(content={'tts_path': 'http://127.0.0.1:8888/' + filename})

@app.post("/inference")
async def inference(target_video: UploadFile = File(...), source_audio: UploadFile = File(...)):
    pass

    audio_path = None

    return JSONResponse(content={'audio_path': 'http://127.0.0.1:8887/' + audio_path})



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8888)
    #uvicorn.run(app, host='0.0.0.0', port=8887)
