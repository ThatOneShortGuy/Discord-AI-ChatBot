import os
import gc

import torch
from flask import Flask, jsonify, request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights
from pprint import pprint
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

app = Flask(__name__)

MODEL_NAME = "OpenAssistant/falcon-7b-sft-mix-2000"

config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)

max_memory = {0: '24GB', 1: '9GB', 'cpu': '20GB'}

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=['GPTNeoXLayer', 'GPTNeoXMLP'])

pprint(device_map)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_size='left')
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=device_map).half()
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map=device_map, load_in_8bit=False, llm_int8_threshold=0, trust_remote_code=True)

model = model.eval()
try:
    model = torch.compile(model, mode='max-autotune', fullgraph=True)
except Exception as e:
    print("Could not compile model:", e)

@app.route('/generate', methods=['POST'])
def generate():
    content = request.json
    inp = content.get('text', '')
    input_ids = tokenizer.encode(inp, return_tensors="pt")
    init_length = input_ids.shape[1]
    input_ids = input_ids.to(model.device)
    output_sequence = model.generate(
            input_ids,
            max_length=2048,
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
        output_sequence = tokenizer.encode(inp, return_tensors="pt")
        init_length = output_sequence.shape[1]
        if init_length > 2048:
            yield f"Input is too long. It has {init_length} tokens, but the maximum is {2048} tokens. Please shorten the input and try again." # 2048 in f-string because increased visibility
            return
        print(f'Input length: {init_length} tokens')
        output_sequence = output_sequence.to(model.device)
        for _ in range(2048-init_length):
            output_sequence = model.generate(
                output_sequence,
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
    app.run(host='0.0.0.0', port=5000)