import gc
import os
import sys
import time
from configparser import ConfigParser
from pprint import pprint
from threading import Thread
from typing import Union

from flask import Flask, Response, jsonify, request

config = ConfigParser()
config.read('config.ini')

profile = sys.argv[1] if len(sys.argv) > 1 else 'default'
os.system("")

if profile not in config:
    print(f'Profile "\033[31m{profile}\033[0m" not found in config.ini')
    print('Please make the profile using `\033[33mpython3 makeConfig.py <profile>\033[0m` or edit config.ini manually.')
    sys.exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if config[profile]['TRANSFORMERS_CACHE']:
    os.environ['TRANSFORMERS_CACHE'] = config[profile]['TRANSFORMERS_CACHE']

from ModelHandler import ModelHandler

model = ModelHandler(config[profile]['language_model'], 'exl2', revision='3_0', max_context_size=2**14+2**12)
model.load_model()

app = Flask(__name__)

isgen = False

@app.route('/generate', methods=['POST'])
def generate():
    global isgen
    content = request.json
    inp: str = content.get('text', '') # type: ignore
    max_tokens = content.get('max_tokens', model.max_context_size) # type: ignore
    max_new_tokens: Union[int, None] = content.get('max_new_tokens', None) # type: ignore
    peft_model = content.get('peft_model', '') # type: ignore

    while isgen:
        time.sleep(1)

    isgen = True
    inp_size = len(model.tokenize(inp))

    if max_new_tokens:
        max_tokens = inp_size + max_new_tokens

    if inp_size > max_tokens:
        isgen = False
        return jsonify({'generated_text': f'Input too long ({inp_size} > {max_tokens})'})
    output = model(inp, max_tokens=max_tokens, stop=['<|im_start|>', '<|im_end|>'])
    isgen = False

    return jsonify({'generated_text': output})

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    global isgen
    gc.collect()
    content = request.json
    inp: str = content.get('text', '') # type: ignore
    max_tokens: int = content.get('max_tokens', model.max_context_size) # type: ignore
    max_new_tokens: Union[int, None] = content.get('max_new_tokens', None) # type: ignore
    stream_all = content.get('stream_all', True) # type: ignore

    for _ in range(35):
        if not isgen:
            break
        time.sleep(1)
        
    isgen = True

    inp_size = len(model.tokenize(inp))

    if max_new_tokens:
        max_tokens = inp_size + max_new_tokens

    # Stream the generated output
    def generate():
        global model, isgen

        print(f'Input tokens: {inp_size}')
        if inp_size > max_tokens:
            yield f'Input too long ({inp_size} > {max_tokens})'
            isgen = False
            return
        
        print(f'max_new_tokens: {max_tokens-inp_size}')
        
        gen = model(inp, max_tokens=max_tokens, stop=['<|im_start|>', '<|im_end|>'], stream=True)
        if stream_all:
            yield f"Generating from {inp_size} tokens... This may take a while."
        text = ''
        for g in gen:
            isgen = False
            if stream_all:
                text += g
                yield text
            else:
                yield g
            isgen = True
        isgen = False

    return Response(generate(), mimetype='text/plain')

@app.route('/embeddings', methods=['POST'])
def embeddings():
    content = request.json
    inp = content.get('text', '') # type: ignore

    output = model.embed(inp)

    return jsonify({'embeddings': output})

@app.route('/model_info', methods=['GET'])
def model_info():
    '''
    Returns the model's information such as the model's name, max num tokens.
    Also returns some of the formatting information such as the start and end token.
    '''
    return jsonify({
        'model_name': model.model_name,
        'max_num_tokens': model.max_context_size,
        'start_token': model.get_bos(),
        'end_token': model.get_eos(),
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=config[profile]['model_server_port'], # type: ignore
        debug=False,
        threaded=True
    )