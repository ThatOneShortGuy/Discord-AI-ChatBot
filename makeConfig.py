from configparser import ConfigParser
import os

config = ConfigParser()
config.read('config.ini')

def makeConfig(profile='DEFAULT'):
    try:
        token = config[profile]['token']
    except KeyError:
        token = input('Enter your token:\n')
        cache_token = input('Cache token for future use? (y/n)\n').lower()
        if cache_token == 'y':
            config[profile]['token'] = token
            with open('config.ini', 'w') as f:
                config.write(f)
    if 'model_server_ip' in config[profile]:
        return
    use_defaults = input('Use default values? (y/n)\n').lower()
    if use_defaults == 'y':
        config[profile]['CUDA_VISIBLE_DEVICES'] = '0'
        config[profile]['language_model'] = 'OpenAssistant/falcon-7b-sft-mix-2000'
        config[profile]['image_gen_model'] = 'stabilityai/stable-diffusion-2-1'
        config[profile]['image_description_ip'] = '127.0.0.1'
        config[profile]['image_description_port'] = '5002'
        config[profile]['model_server_ip'] = '127.0.0.1'
        config[profile]['model_server_port'] = '5000'
        config[profile]['image_gen_ip'] = '127.0.0.1'
        config[profile]['image_gen_port'] = '5001'

        # If os is windows, use the default cache path
        if os.name == 'nt':
            TRANSFORMERS_CACHE = ''
        else:
            TRANSFORMERS_CACHE = '/tmp/transformers_cache'
        config[profile]['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE

        with open('config.ini', 'w') as f:
            config.write(f)
        return
    
    try:
        CUDA_VISIBLE_DEVICES = config[profile]['CUDA_VISIBLE_DEVICES']
    except KeyError:
        CUDA_VISIBLE_DEVICES = input('Enter the CUDA_VISIBLE_DEVICES value (e.g. "0" or "0,1"):\n')
        config[profile]['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        language_model = config[profile]['language_model']
    except KeyError:
        language_model = input('Enter the Hugging Face path to the language model (e.g. OpenAssistant/falcon-7b-sft-mix-2000):\n')
        config[profile]['language_model'] = language_model
        with open('config.ini', 'w') as f:
            config.write(f)

    try:
        image_gen_model = config[profile]['image_gen_model']
    except KeyError:
        image_gen_model = input('Enter the Hugging Face path to the image generation model (e.g. stabilityai/stable-diffusion-2-1):\n')
        config[profile]['image_gen_model'] = image_gen_model
        with open('config.ini', 'w') as f:
            config.write(f)

    try:
        image_description_ip = config[profile]['image_description_ip']
    except KeyError:
        image_description_ip = input('Enter the IP of the image description server:\n')
        config[profile]['image_description_ip'] = image_description_ip
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        image_description_port = config[profile]['image_description_port']
    except KeyError:
        image_description_port = input('Enter the port of the image description server:\n')
        config[profile]['image_description_port'] = image_description_port
        with open('config.ini', 'w') as f:
            config.write(f)

    try:
        TRANSFORMERS_CACHE = config[profile]['TRANSFORMERS_CACHE']
    except KeyError:
        TRANSFORMERS_CACHE = input('Enter the path to the transformers cache:\n')
        config[profile]['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        model_server_ip = config[profile]['model_server_ip']
    except KeyError:
        model_server_ip = input('Enter the IP of the model server:\n')
        config[profile]['model_server_ip'] = model_server_ip
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        model_server_port = config[profile]['model_server_port']
    except KeyError:
        model_server_port = input('Enter the port of the model server:\n')
        config[profile]['model_server_port'] = model_server_port
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        image_gen_ip = config[profile]['image_gen_ip']
    except KeyError:
        image_gen_ip = input('Enter the IP of the image generation server:\n')
        config[profile]['image_gen_ip'] = image_gen_ip
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        image_gen_port = config[profile]['image_gen_port']
    except KeyError:
        image_gen_port = input('Enter the port of the image generation server:\n')
        config[profile]['image_gen_port'] = image_gen_port
        with open('config.ini', 'w') as f:
            config.write(f)
