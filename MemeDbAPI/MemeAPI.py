# https://epsilla-inc.gitbook.io/epsilladb/api-reference      API Reference

from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from pyepsilla import vectordb
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device) # type: ignore

def get_image_embedding(image: np.ndarray) -> list[float]:
    with torch.no_grad():
        inputs = processor(text='', images=image, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        outputs = model(**inputs)
    return outputs.image_embeds.cpu().numpy()[0].tolist()

def get_images_embedding(images: list[np.ndarray]) -> list[list[float]]:
    with torch.no_grad():
        inputs = processor(text='', images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        outputs = model(**inputs)
    return outputs.image_embeds.cpu().numpy().tolist()

def get_text_embedding(text: str) -> list[float]:
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        outputs = model.get_text_features(**inputs) # type: ignore
    return outputs.cpu().numpy()[0].tolist()

class Database:
    def __init__(self,
                 table_name: str,
                 host: str = '192.168.1.6',
                 port: str = '8888',
                 img_processing_function: Callable[[np.ndarray], list[float]] = get_image_embedding,
                 str_processing_function: Callable[[str], list[float]] = get_text_embedding):
        self.client = vectordb.Client(host=host, port=port)
        self.host = host
        self.port = port
        self.client.load_db(db_name='MyDB', db_path='/tmp/epsilla')
        self.client.use_db('MyDB')
        self.table_name = table_name
        self.img_processing_function = img_processing_function
        self.str_processing_function = str_processing_function
    
    def insert(self, records: list[dict]):
        return self.client.insert(
            table_name=self.table_name,
            records=records
        )
    
    def format_img(self, img: Union[np.ndarray, Image.Image]) -> list[float]:
        if isinstance(img, Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if self.img_processing_function:
            return self.img_processing_function(img)
        
        return img.tolist()

    def format_text(self, text: str) -> list[float]:
        return self.str_processing_function(text)
    
    def query(self, query_value: Union[str, np.ndarray, Image.Image], limit: int = 1, with_distance: bool = True) -> list[dict]:
        if isinstance(query_value, str):
            return self.query_text(query_value, limit=limit, with_distance=with_distance)
        elif isinstance(query_value, np.ndarray) or isinstance(query_value, Image.Image):
            return self.query_img(query_value, limit=limit, with_distance=with_distance)
        else:
            raise Exception(f'Invalid query type: {type(query_value)}')

    def query_text(self, query_text: str, limit: int = 1, with_distance: bool = True) -> list[dict]:
        query_value: list[float] = self.format_text(query_text)
        return self._query(query_value=query_value, limit=limit, with_distance=with_distance)
    
    def query_img(self, query_img: Union[np.ndarray, Image.Image], limit: int = 1, with_distance: bool = True) -> list[dict]:
        query_value: list[float] = self.format_img(query_img)
        return self._query(query_value=query_value, limit=limit, with_distance=with_distance)
    
    def _query(self, query_value: list[float], limit: int = 1, with_distance: bool = True) -> list[dict]:
        query_field = 'PixelVec'

        status, response = self.client.query(
            table_name=self.table_name,
            query_field=query_field,
            query_vector=query_value,
            limit=limit,
            with_distance=with_distance
        )

        print(f"\033[31m{response['message']}\033[0m")
        if status != 200:
            raise Exception(f'Meme Query failed with status {status}')
        
        return response['result']