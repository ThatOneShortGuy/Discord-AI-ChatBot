import io
import json
import os
import socket

import requests
from PIL import Image

os.system("")

SERVER_IP = (([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0]
URL = f'http://{SERVER_IP}:5001/generate'

def generate(prompt, neg_prompt='', img_type='normal', width=768, height=768, num_inference_steps=69):
    data = json.dumps({'prompt': prompt, 'height': height, 'width': width,
                       'type': img_type, 'num_inference_steps': num_inference_steps,
                       'neg_prompt': neg_prompt, 'save': False})
    headers = {'content-type': 'application/json'}
    response = requests.post(URL, data=data, headers=headers)
    content = response.content
    image = io.BytesIO(content)
    image = Image.open(image)
    return image

if __name__ == '__main__':
    img = generate('Mona Lisa in the style of Picaso')
    img.show()