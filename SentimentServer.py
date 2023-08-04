import os
import sys
from configparser import ConfigParser

import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, jsonify, request

config = ConfigParser()
config.read('config.ini')

profile = sys.argv[1] if len(sys.argv) > 1 else 'default'
os.system("")

if profile not in config:
    print(f'Profile "\033[31m{profile}\033[0m" not found in config.ini')
    print('Please make the profile using `\033[33mpython3 makeConfig.py <profile>\033[0m` or edit config.ini manually.')
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = config[profile]['CUDA_VISIBLE_DEVICES']
if config[profile]['TRANSFORMERS_CACHE']:
    os.environ['TRANSFORMERS_CACHE'] = config[profile]['TRANSFORMERS_CACHE']

if 'sentiment_model' not in config[profile]:
    print(f'Profile "\033[31m{profile}\033[0m" does not have a `\033[31msentiment_model\033[0m` in config.ini')
    print('Please add the `\033[31msentiment_model\033[0m` to the profile using `\033[33mpython3 makeConfig.py <profile>\033[0m` or edit config.ini manually.')
    sys.exit(1)

MODEL_NAME = config[profile]['sentiment_model']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, device_map={'':0})

model = model.eval()

app = Flask(__name__)

@app.route('/sentiment', methods=['POST'])
def sentiment():
    content = request.json
    text = content.get('text', '')
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    outputs = model(**inputs)
    outputs = outputs[0][0].detach().numpy()
    probs = softmax(outputs)
    return jsonify({
        'negative': float(probs[0]),
        'neutral': float(probs[1]),
        'positive': float(probs[2])
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=config[profile]['sentiment_port'],
        debug=False,
        threaded=True
    )