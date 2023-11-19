import gc
import os
import sys
from configparser import ConfigParser
from pprint import pprint
from threading import Thread

from flask import Flask, Response, jsonify, request
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer

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

app = Flask(__name__)

MODEL_NAME = "TheBloke/CodeLlama-13B-oasst-sft-v10-GGUF"

tokenizer = AutoTokenizer.from_pretrained(config[profile]['language_model'])

print(f'Loading model: {MODEL_NAME}')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    model_file="codellama-13b-oasst-sft-v10.Q4_K_M.gguf",
    model_type="llama2",
    gpu_layers=1300,
    context_length=8192,
)

print(f'Max num tokens: {model.context_length}')

@app.route('/generate', methods=['POST'])
def generate():
    content = request.json
    inp = content.get('text', '') # type: ignore
    max_tokens = content.get('max_tokens', model.context_length) # type: ignore
    peft_model = content.get('peft_model', '') # type: ignore

    inp_size = len(model.tokenize(inp))
    if inp_size > max_tokens:
        return jsonify({'generated_text': f'Input too long ({inp_size} > {max_tokens})'})

    output = model(inp, max_new_tokens=max_tokens-inp_size, stop=[tokenizer.eos_token, tokenizer.bos_token], threads=1, batch_size=512)

    return jsonify({'generated_text': output})

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    gc.collect()
    content = request.json
    inp = content.get('text', '') # type: ignore
    max_tokens = content.get('max_tokens', model.context_length) # type: ignore

    inp_size = len(model.tokenize(inp))

    # Stream the generated output
    def generate():
        global model

        print(f'Input tokens: {inp_size}')
        if inp_size > max_tokens:
            yield f'Input too long ({inp_size} > {max_tokens})'
            return
        
        print(f'max_new_tokens: {max_tokens-inp_size}')
        output = model(inp, max_new_tokens=max_tokens-inp_size, stop=[tokenizer.eos_token, tokenizer.bos_token], threads=1, batch_size=512, stream=True)
        yield f"Generating from {inp_size} tokens... This may take a while."
        text = ''
        for i in output:
            text += i
            yield text

    return Response(generate(), mimetype='text/plain')

@app.route('/embeddings', methods=['POST'])
def embeddings():
    content = request.json
    inp = content.get('text', '') # type: ignore

    output = model.embed(
        inp,
        threads=1,
        batch_size=512,
    )

    return jsonify({'embeddings': output})

@app.route('/model_info', methods=['GET'])
def model_info():
    '''
    Returns the model's information such as the model's name, max num tokens.
    Also returns some of the formatting information such as the start and end token.
    '''
    return jsonify({
        'model_name': MODEL_NAME,
        'max_num_tokens': model.context_length,
        'start_token': tokenizer.bos_token,
        'end_token': tokenizer.eos_token,
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=config[profile]['model_server_port'], # type: ignore
        debug=False,
        threaded=True
    )