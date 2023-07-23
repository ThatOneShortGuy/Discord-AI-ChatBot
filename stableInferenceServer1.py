import gc
import hashlib
import io
import json
import os
import re
from configparser import ConfigParser
import sys

import torch
from diffusers import (DiffusionPipeline, DPMSolverMultistepScheduler,
                       StableDiffusionUpscalePipeline)
from flask import Flask, jsonify, request, send_file
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

config = ConfigParser()
config.read('config.ini')

profile = sys.argv[1] if len(sys.argv) > 1 else 'DEFAULT'

os.environ['TRANSFORMERS_CACHE'] = config[profile]['TRANSFORMERS_CACHE'] if os.path.exists('config.ini') else 'D:\TransformersCache'

os.chdir(os.path.dirname(os.path.realpath(__file__))) # Change or files may be saved in C:\Windows\System32

def init_upscaler():
    global upscaler
    upscaler = StableDiffusionUpscalePipeline.from_pretrained('stabilityai/stable-diffusion-x4-upscaler', torch_dtype=torch.float16).to('cuda:0')

def pipe_type(model_name):
    global pipe
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        custom_pipeline='lpw_stable_diffusion',
        safety_checker=None,
        requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")
    pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    # Workaround for not accepting attention shape using VAE for Flash Attention
    pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
    gc.collect()

def upscale_image(prompt, image, **kwargs):
    global upscaler, pipe
    pipe = pipe.to('cpu')
    img = upscaler(prompt, image, **kwargs).images[0]
    pipe = pipe.to('cuda:0')
    gc.collect()
    torch.cuda.empty_cache()
    return img

pipe_type(config[profile]['image_gen_model'] if os.path.exists('config.ini') else 'stabilityai/stable-diffusion-2-1')

global model_type
model_type = 'normal'

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    global model_type, pipe, upscaler
    content = request.json

    # Get parameters
    prompt = content.get('prompt', '')
    neg_prompt = content.get('neg_prompt', '')
    img_type = content.get('type', 'normal').lower()
    width = int(content.get('width', 768 if img_type == 'normal' else 512))
    height = int(content.get('height', 768 if img_type == 'normal' else 512))
    num_inference_steps = int(content.get('num_inference_steps', 69))
    upscale = bool(content.get('upscale', False))
    save = bool(content.get('save', True))

    # Load upscaler if needed
    if upscale:
        init_upscaler()

    # Switch model if needed
    if img_type == 'waifu' and model_type != 'waifu':
        pipe_type('hakurei/waifu-diffusion')
        model_type = 'waifu'
        print('Switched to waifu model')
    elif img_type == 'normal' and model_type != 'normal':
        pipe_type(config[profile]['image_gen_model'] if os.path.exists('config.ini') else 'stabilityai/stable-diffusion-2-1')
        model_type = 'normal'
        print('Switched to normal model')

    # Filter prompt
    filtered_prompt = re.sub(r'([^\w\s]|\d)', '', prompt)
    filtered_prompt = re.sub(r'\s+', ' ', filtered_prompt).strip()
    folder = os.path.join('images', filtered_prompt[:50])
    print(f'Prompt: {filtered_prompt}, Width: {width}, Height: {height}, Type: {img_type}, Steps: {num_inference_steps}')

    # Generate image
    image = pipe(prompt, width=width, height=height, negative_prompt=neg_prompt,
                 num_inference_steps=num_inference_steps).images[0]

    # Upscale image
    image = upscale_image(prompt, image, negative_prompt=neg_prompt, num_inference_steps=15) if upscale else image
                     
    # Save image to disk
    if save:
        image_data = image.tobytes()
        filename = hashlib.md5(image_data[:2048]).hexdigest()+'.jpg'
        if not os.path.exists(folder):
            os.makedirs(folder)
        if os.path.exists(os.path.join(folder, 'prompts.txt')):
            with open(os.path.join(folder, 'prompts.txt'), 'r') as f:
                contents = f.read()
                contents = re.sub(r'\}\n\{', ',\n', contents)
                contents = json.loads(contents)
            contents[filename] = content
        else:
            contents = {filename: content}
        with open(os.path.join(folder, 'prompts.txt'), 'w') as f:
            f.write(json.dumps(contents, indent=2))
        filename = os.path.join(folder, filename)
        image.save(filename, quality=95, subsampling=0)
        return send_file(filename, mimetype='image/jpg')
    else:
        print('Not saving image')
        file = io.BytesIO()
        image.save(file, format='png')
        file.seek(0)
        return send_file(file, mimetype='image/png')

if __name__ == '__main__':
    app.run(host=config[profile]['image_gen_ip'], port=config[profile]['image_gen_port'], debug=False)