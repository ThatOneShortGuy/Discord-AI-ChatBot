import io
import json
import os
import socket
import time
from configparser import ConfigParser
import sys

import requests
from PIL import Image

os.system("")

config = ConfigParser()
config.read('config.ini')

profile = sys.argv[1] if len(sys.argv) > 1 else 'default'

SERVER_IP = config.get(profile, 'image_gen_ip')
PORT = config.get(profile, 'image_gen_port')
URL = f'http://{SERVER_IP}:{PORT}/generate'

print("Server IP:", SERVER_IP)
print("Port:", PORT)

def start_server():
    if os.name == 'nt':
        os.system("start python StableDiffusionAPI/stableInferenceServer.py")
    else:
        os.system("python3 ./StableDiffusionAPI/stableInferenceServer.py &")

def get_response(data, headers):
    err = False
    while True:
        try:
            response = requests.post(URL, data=data, headers=headers)
            return response
        except requests.exceptions.ConnectionError:
            print('Waiting for server to start...')
            if not err:
                if config[profile]['image_description_ip'] != '127.0.0.1':
                    print(f'Could not start image description server because remote server is not running on {config[profile]["image_description_ip"]}:{config[profile]["image_description_port"]}')
                    return 'Image description failed'
                start_server()
                err = True
            time.sleep(5)

def generate(prompt, neg_prompt='', img_type='normal', width=768, height=768, num_inference_steps=69):
    data = json.dumps({'prompt': prompt, 'height': height, 'width': width,
                       'type': img_type, 'num_inference_steps': num_inference_steps,
                       'neg_prompt': neg_prompt, 'save': False})
    headers = {'content-type': 'application/json'}
    response = get_response(data, headers)
    if isinstance(response, str):
        return response
    content = response.content
    image = io.BytesIO(content)
    image = Image.open(image)
    return image

if __name__ == '__main__':
    img = generate('Mona Lisa in the style of Picaso')
    img.show()