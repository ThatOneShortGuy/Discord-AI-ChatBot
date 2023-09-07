import gc
import os
import sys
from configparser import ConfigParser
from pprint import pprint
from threading import Thread

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from flask import Flask, Response, jsonify, request
from peft import PeftModel
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)

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

app = Flask(__name__)

MODEL_NAME = config[profile]['language_model']

model_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

max_memory = {0: '19000MB', 1: '8000MB', 'cpu': '59GB'}

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.float16, trust_remote_code=True)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=['LlamaDecoderLayer'])

print('Device map:')
pprint(device_map)
print('Using model:', MODEL_NAME)


current_model = '16bit'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
print(f'Max token length: {tokenizer.model_max_length}')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True).half().eval()

# model = PeftModel.from_pretrained(model, "ThatOneShortGuy/MusicalFalcon", is_trainable=False)

def load_model_as_16bit():
    global current_model, model
    if current_model == '16bit':
        return model
    del model
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    print('Loading model as 16bit')
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True).half().eval()
    current_model = '16bit'

def load_model_as_8bit():
    global current_model, model
    if current_model == '8bit':
        return model
    del model
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    print('Loading model as 8bit')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device_map,
        load_in_8bit=True, llm_int8_threshold=6,
        trust_remote_code=True,
        llm_int8_enable_fp32_cpu_offload=True
        )
    model = model.eval()
    current_model = '8bit'

@app.route('/generate', methods=['POST'])
def generate():
    content = request.json
    inp = content.get('text', '')
    max_tokens = content.get('max_tokens', tokenizer.model_max_length)
    peft_model = content.get('peft_model', '')

    if peft_model:
        peft_model = PeftModel.from_pretrained(model, peft_model, is_trainable=False)
    else:
        peft_model = model
    peft_model = peft_model.eval()

    input_ids = tokenizer.encode(inp, return_tensors="pt")
    init_length = input_ids.shape[1]
    input_ids = input_ids.to(model.device)
    output_sequence = peft_model.generate(
            inputs=input_ids,
            max_new_tokens=max_tokens,
            top_k=69,
            top_p=.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            temperature=.95,
            use_cache=True,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    decoded_output = tokenizer.decode(output_sequence[0][init_length:-1])

    return jsonify({'generated_text': decoded_output})

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    content = request.json
    inp = content.get('text', '')
    max_tokens = content.get('max_tokens', tokenizer.model_max_length)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Stream the generated output
    def generate():
        global model

        output_sequence = tokenizer.encode(inp, return_tensors="pt", padding=True)
        init_length = output_sequence.shape[1]
        if init_length > max_tokens:
            yield f"Input is too long. It has {init_length} tokens, but the maximum is {max_tokens} tokens. Please shorten the input and try again."
            return

        print(f'Input length: {init_length} tokens')
        if init_length > 2900:
            yield f'Loading 8-bit model...\nPlease wait...'
            load_model_as_8bit()
        else:
            yield f'Loading 16-bit model...\nPlease wait...'
            load_model_as_16bit()

        peft_model = content.get('peft_model', '')
        if peft_model:
            yield f'Loading PEFT model: {peft_model}\nPlease wait...'
            peft_model = PeftModel.from_pretrained(model, peft_model, is_trainable=False)
        else:
            peft_model = model
        peft_model = peft_model.eval()
        output_sequence = output_sequence.to(model.device)

        model_kwargs = dict(
            input_ids=output_sequence,
            streamer=streamer,
            max_new_tokens=max_tokens-init_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=20,
            temperature=.7,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

        t = Thread(target=peft_model.generate, kwargs=model_kwargs)
        t.start()

        text = ''
        for new_token in streamer:
            text += new_token
            yield text
        
        t.join()
        del output_sequence
        
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    return Response(generate(), mimetype='text/plain')

@app.route('/model_info', methods=['GET'])
def model_info():
    '''
    Returns the model's information such as the model's name, max num tokens.
    Also returns some of the formatting information such as the start and end token.
    '''
    return jsonify({
        'model_name': MODEL_NAME,
        'max_num_tokens': tokenizer.model_max_length,
        'start_token': tokenizer.bos_token,
        'end_token': tokenizer.eos_token,
    })

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=config[profile]['model_server_port'],
        debug=False,
        threaded=True
    )