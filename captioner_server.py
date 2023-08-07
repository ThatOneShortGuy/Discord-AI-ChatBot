import os
import gc
import requests

import torch
from flask import Flask, jsonify, request, Response
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

MODEL_NAME = "Salesforce/blip-image-captioning-large"

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, device_map={'': 1}, torch_dtype=torch.float16)

model = model.eval()

try:
    model = torch.compile(model, mode='max-autotune', fullgraph=True)
except Exception as e:
    print("Could not compile model:", e)

@app.route('/describe', methods=['POST'])
def describe():
    content = request.json
    img = content.get('image', '')
    text = content.get('text', '')
    print(f'{img=}')
    print(f'{text=}')
    try:
        raw_image = Image.open(img).convert('RGB')
    except OSError:
        raw_image = Image.open(requests.get(img, stream=True).raw).convert('RGB')
    inputs = processor(raw_image, text, return_tensors="pt").to(model.device, torch.float16)
    out = model.generate(**inputs)
    desc = processor.decode(out[0], skip_special_tokens=True)
    print(f'{desc=}')
    return jsonify(desc)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)