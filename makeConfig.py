from configparser import ConfigParser
import os
import sys

config = ConfigParser()
config.read('config.ini')

DEFAULTS = {
    'CUDA_VISIBLE_DEVICES': '0',
    'language_model': 'OpenAssistant/falcon-7b-sft-mix-2000',
    'image_gen_model': 'stabilityai/stable-diffusion-2-1',
    'sentiment_model': 'cardiffnlp/twitter-roberta-base-sentiment',
    'image_description_ip': '127.0.0.1',
    'image_description_port': '5002',
    'model_server_ip': '127.0.0.1',
    'model_server_port': '5000',
    'image_gen_ip': '127.0.0.1',
    'image_gen_port': '5001',
    'sentiment_ip': '127.0.0.1',
    'sentiment_port': '5003',
    'MemeDB_ip': '127.0.0.1',
    'MemeDB_port': '8888',
    'meme_channel_id': '744987855335456900',
}

def makeConfig(profile='default'):
    if profile not in config:
        config[profile] = {}

    # Ask for Discord token
    try:
        token = config[profile]['token']
    except KeyError:
        token = input('Enter your token:\n')
        cache_token = input('Cache token for future use? (y/n)\n').lower()
        if cache_token == 'y':
            config[profile]['token'] = token
            with open('config.ini', 'w') as f:
                config.write(f)
    
    # Test to see if we need to ask user for defaults
    if 'model_server_ip' not in config[profile]:
        use_defaults = input('Use default values? (y/n)\n').lower()
    else:
        use_defaults = 'n'
    if use_defaults == 'y':
        # Use the default values
        for key, value in DEFAULTS.items():
            config[profile][key] = value

        # Use the default Cache
        TRANSFORMERS_CACHE = ''
        config[profile]['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE

        with open('config.ini', 'w') as f:
            config.write(f)
        return

    try:
        CUDA_VISIBLE_DEVICES = config[profile]['CUDA_VISIBLE_DEVICES']
    except KeyError:
        CUDA_VISIBLE_DEVICES = input(f'Enter the CUDA_VISIBLE_DEVICES value (e.g. "0" or "0,1") (default: {DEFAULTS["CUDA_VISIBLE_DEVICES"]}):\n')
        if CUDA_VISIBLE_DEVICES == '':
            CUDA_VISIBLE_DEVICES = DEFAULTS['CUDA_VISIBLE_DEVICES']
        config[profile]['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        language_model = config[profile]['language_model']
    except KeyError:
        language_model = input(f'Enter the Hugging Face path to the language model (default: {DEFAULTS["language_model"]}):\n')
        if language_model == '':
            language_model = DEFAULTS['language_model']
        config[profile]['language_model'] = language_model
        with open('config.ini', 'w') as f:
            config.write(f)

    try:
        image_gen_model = config[profile]['image_gen_model']
    except KeyError:
        image_gen_model = input(f'Enter the Hugging Face path to the image generation model (default: {DEFAULTS["image_gen_model"]}):\n')
        if image_gen_model == '':
            image_gen_model = DEFAULTS['image_gen_model']
        config[profile]['image_gen_model'] = image_gen_model
        with open('config.ini', 'w') as f:
            config.write(f)

    try:
        sentiment_model = config[profile]['sentiment_model']
    except KeyError:
        sentiment_model = input(f'Enter the Hugging Face path to the sentiment model (default: {DEFAULTS["sentiment_model"]}):\n')
        if sentiment_model == '':
            sentiment_model = DEFAULTS['sentiment_model']
        config[profile]['sentiment_model'] = sentiment_model
        with open('config.ini', 'w') as f:
            config.write(f)

    try:
        image_description_ip = config[profile]['image_description_ip']
    except KeyError:
        image_description_ip = input(f'Enter the IP of the image description server (default: {DEFAULTS["image_description_ip"]}):\n')
        if image_description_ip == '':
            image_description_ip = DEFAULTS['image_description_ip']
        config[profile]['image_description_ip'] = image_description_ip
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        image_description_port = config[profile]['image_description_port']
    except KeyError:
        image_description_port = input(f'Enter the port of the image description server (default: {DEFAULTS["image_description_port"]}):\n')
        if image_description_port == '':
            image_description_port = DEFAULTS['image_description_port']
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
        model_server_ip = input(f'Enter the IP of the model server (default: {DEFAULTS["model_server_ip"]}):\n')
        if model_server_ip == '':
            model_server_ip = DEFAULTS['model_server_ip']
        config[profile]['model_server_ip'] = model_server_ip
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        model_server_port = config[profile]['model_server_port']
    except KeyError:
        model_server_port = input(f'Enter the port of the model server (default: {DEFAULTS["model_server_port"]}):\n')
        if model_server_port == '':
            model_server_port = DEFAULTS['model_server_port']
        config[profile]['model_server_port'] = model_server_port
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        image_gen_ip = config[profile]['image_gen_ip']
    except KeyError:
        image_gen_ip = input(f'Enter the IP of the image generation server (default: {DEFAULTS["image_gen_ip"]}):\n')
        if image_gen_ip == '':
            image_gen_ip = DEFAULTS['image_gen_ip']
        config[profile]['image_gen_ip'] = image_gen_ip
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        image_gen_port = config[profile]['image_gen_port']
    except KeyError:
        image_gen_port = input(f'Enter the port of the image generation server (default: {DEFAULTS["image_gen_port"]}):\n')
        if image_gen_port == '':
            image_gen_port = DEFAULTS['image_gen_port']
        config[profile]['image_gen_port'] = image_gen_port
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        sentiment_ip = config[profile]['sentiment_ip']
    except KeyError:
        sentiment_ip = input(f'Enter the IP of the sentiment server (default: {DEFAULTS["sentiment_ip"]}):\n')
        if sentiment_ip == '':
            sentiment_ip = DEFAULTS['sentiment_ip']
        config[profile]['sentiment_ip'] = sentiment_ip
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        sentiment_port = config[profile]['sentiment_port']
    except KeyError:
        sentiment_port = input(f'Enter the port of the sentiment server (default: {DEFAULTS["sentiment_port"]}):\n')
        if sentiment_port == '':
            sentiment_port = DEFAULTS['sentiment_port']
        config[profile]['sentiment_port'] = sentiment_port
        with open('config.ini', 'w') as f:
            config.write(f)
    
    try:
        meme_db_ip = config[profile]['MemeDB_ip']
    except KeyError:
        meme_db_ip = input(f'Enter the IP of the MemeDB server (default: {DEFAULTS["MemeDB_ip"]}):\n')
        if meme_db_ip == '':
            meme_db_ip = DEFAULTS['MemeDB_ip']
        config[profile]['MemeDB_ip'] = meme_db_ip
        with open('config.ini', 'w') as f:
            config.write(f)
        
    try:
        meme_db_port = config[profile]['MemeDB_port']
    except KeyError:
        meme_db_port = input(f'Enter the port of the MemeDB server (default: {DEFAULTS["MemeDB_port"]}):\n')
        if meme_db_port == '':
            meme_db_port = DEFAULTS['MemeDB_port']
        config[profile]['MemeDB_port'] = meme_db_port
        with open('config.ini', 'w') as f:
            config.write(f)

    try:
        meme_channel_id = config[profile]['meme_channel_id']
    except KeyError:
        meme_channel_id = input(f'Enter the channel ID of the meme channel:\n')
        if meme_channel_id == '':
            meme_channel_id = DEFAULTS['meme_channel_id']
        config[profile]['meme_channel_id'] = meme_channel_id
        with open('config.ini', 'w') as f:
            config.write(f)

makeConfig(sys.argv[1] if len(sys.argv) > 1 else 'default')