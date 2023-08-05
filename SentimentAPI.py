import json
import os
import time
from configparser import ConfigParser
import sys

import requests

os.system("")

config = ConfigParser()
config.read('config.ini')

profile = sys.argv[1] if len(sys.argv) == 2 else 'default'

SERVER_IP = config.get(profile, 'sentiment_ip')
PORT = config.get(profile, 'sentiment_port')
URL = f'http://{SERVER_IP}:{PORT}/sentiment'

def get_sentiment(text) -> dict:
    data = json.dumps({'text': text})
    headers = {'content-type': 'application/json'}
    response = requests.post(URL, data=data, headers=headers)
    return response.json()

def get_most_likely_sentiment(text) -> str:
    sentiment = get_sentiment(text)
    sentiment = sorted(sentiment.items(), key=lambda x: x[1], reverse=True)
    return sentiment[0][0]

def is_positive_sentiment(text) -> bool:
    sentiment = get_sentiment(text)
    return sentiment['positive'] > sentiment['negative']

if __name__ == '__main__':
    sentiment = get_most_likely_sentiment('I am sad')
    print(sentiment)