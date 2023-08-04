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

def generate(prompt, neg_prompt='', img_type='normal', width=768, height=768, num_inference_steps=69):
    data = json.dumps({'prompt': prompt, 'height': height, 'width': width,
                       'type': img_type, 'num_inference_steps': num_inference_steps,
                       'neg_prompt': neg_prompt, 'save': False})
    headers = {'content-type': 'application/json'}
    err = False
    # response = requests.post(URL, data=data, headers=headers)
    while True:
        try:
            response = requests.post(URL, data=data, headers=headers)
            break
        except requests.exceptions.ConnectionError:
            print('Waiting for server to start...')
            if err:
                time.sleep(3)
                continue
            if os.name == 'nt':
                os.system("start python stableInferenceServer.py")
            else:
                os.system("python3 stableInferenceServer.py &")
            err = True
            time.sleep(5)
    content = response.content
    image = io.BytesIO(content)
    image = Image.open(image)
    return image

if __name__ == '__main__':
    img = generate('Mona Lisa in the style of Picaso')
    img.show()