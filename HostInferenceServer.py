import gc
import os
import re
from configparser import ConfigParser
from pprint import pprint
import sys

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from flask import Flask, Response, jsonify, request
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

max_memory = {0: '20GB', 1: '9GB', 'cpu': '59GB'}

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(model_config, torch_dtype=torch.float16, trust_remote_code=True)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=[''])

print('Device map:')
pprint(device_map)
print('Using model:', MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
print(f'Max token length: {tokenizer.model_max_length}')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True).half()
# model = AutoModelForCausalLM.from_pretrained(
#                                              MODEL_NAME,
#                                              torch_dtype=torch.float16,
#                                              device_map=device_map,
#                                              load_in_8bit=True, llm_int8_threshold=0,
#                                              trust_remote_code=True,
#                                              llm_int8_enable_fp32_cpu_offload=True)

# model = PeftModel.from_pretrained(model, "ThatOneShortGuy/MusicalFalcon", is_trainable=False)

model = model.eval()
try:
    model = torch.compile(model, mode='max-autotune', fullgraph=True)
except Exception as e:
    print("Could not compile model:", e)


# batch = tokenizer("Hello, my dog is cute", return_tensors="pt", padding=True, truncation=True).to(model.device)
# tensors = model.generate(inputs=batch.input_ids, max_new_tokens=69, num_return_sequences=1)
# print(f'Tensors: {tensors}')
# print(tokenizer.decode(tensors.detach()[0].tolist())) # Warmup

@app.route('/generate', methods=['POST'])
def generate():
    content = request.json
    inp = content.get('text', '')
    input_ids = tokenizer.encode(inp, return_tensors="pt")
    init_length = input_ids.shape[1]
    input_ids = input_ids.to(model.device)
    output_sequence = model.generate(
            input_ids,
            max_length=tokenizer.model_max_length,
            do_sample=True,
            top_k=69,
            top_p=.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            temperature=.95,
            use_cache=True,
            no_repeat_ngram_size=3,
            early_stopping=True)

    decoded_output = tokenizer.decode(output_sequence[0][init_length:-1])

    return jsonify({'generated_text': decoded_output})

@app.route('/generate_stream', methods=['POST'])
def generate_stream():
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    content = request.json
    inp = content.get('text', '')
    # Stream the generated output
    def generate():
        output_sequence = tokenizer.encode(inp, return_tensors="pt", padding=True)
        init_length = output_sequence.shape[1]
        if init_length > 2048:
            yield f"Input is too long. It has {init_length} tokens, but the maximum is {2048} tokens. Please shorten the input and try again." # 2048 in f-string because increased visibility
            return
        print(f'Input length: {init_length} tokens')
        output_sequence = output_sequence.to(model.device)
        for _ in range(2048-init_length):
            output_sequence = model.generate(
                inputs=output_sequence,
                do_sample=True,
                top_k=69,
                top_p=.7,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                temperature=.95,
                use_cache=True,
                no_repeat_ngram_size=3,
                max_new_tokens=1,)
            if output_sequence[0][-1] == tokenizer.eos_token_id:
                break
            decoded_output = tokenizer.decode(output_sequence[0][init_length:].tolist())
            yield decoded_output
    return Response(generate(), mimetype='text/plain')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=config[profile]['model_server_port'],
        debug=False,
        threaded=True
    )