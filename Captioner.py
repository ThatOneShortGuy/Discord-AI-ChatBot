import sys
from configparser import ConfigParser

import requests

from makeConfig import makeConfig

profile = sys.argv[1] if len(sys.argv) > 1 else 'default'

makeConfig(profile)

config = ConfigParser()
config.read('config.ini')

def describe_image(image):
    try:
        return requests.post(f'http://{config[profile]["image_description_ip"]}:{config[profile]["image_description_port"]}/describe', json={'image': image}).json()
    except Exception as e:
        print('Could not describe image:', e)
        return 'Image description failed'