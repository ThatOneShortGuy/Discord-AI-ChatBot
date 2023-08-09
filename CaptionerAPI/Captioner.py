import os
import sys
import time
from configparser import ConfigParser

import requests

from makeConfig import makeConfig

profile = sys.argv[1] if len(sys.argv) > 1 else 'default'

makeConfig(profile)

config = ConfigParser()
config.read('config.ini')

def start_server():
    if os.name == 'nt':
        os.system("start python CaptionerAPI/captioner_server.py")
    else:
        os.system("python3 ./CaptionerAPI/captioner_server.py &")

def describe_image(image):
    err = False
    while True:
        try:
            return requests.post(f'http://{config[profile]["image_description_ip"]}:{config[profile]["image_description_port"]}/describe', json={'image': image}).json()
        except requests.exceptions.ConnectionError:
            if not err:
                if config[profile]['image_description_ip'] != '127.0.0.1':
                    print(f'Could not start image description server because remote server is not running on {config[profile]["image_description_ip"]}:{config[profile]["image_description_port"]}')
                    return 'Image description failed'
                start_server()
                err = True
            time.sleep(5)
        except Exception as e:
            print('Could not describe image:', e)
            return 'Image description failed'