# https://epsilla-inc.gitbook.io/epsilladb/api-reference      API Reference

from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from pyepsilla import vectordb
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device) # type: ignore

def get_image_embedding(image: np.ndarray) -> list[float]:
    with torch.no_grad():
        inputs = processor(text='', images=image, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        outputs = model(**inputs)
    return outputs.image_embeds.cpu().numpy()[0].tolist()

class Database:
    def __init__(self, table_name: str, host: str = '192.168.1.16', port: str = '8888', processing_function: Optional[Callable[[np.ndarray], list[float]]] = None):
        self.client = vectordb.Client(host=host, port=port)
        self.host = host
        self.port = port
        self.client.load_db(db_name='MyDB', db_path='/tmp/epsilla')
        self.client.use_db('MyDB')
        self.table_name = table_name
        self.processing_function = processing_function
    
    def insert(self, records: list[dict]):
        return self.client.insert(
            table_name=self.table_name,
            records=records
        )
    
    def format_img(self, img: Union[np.ndarray, Image.Image]) -> list[float]:
        if isinstance(img, Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if self.processing_function:
            return self.processing_function(img)
        
        if img.shape[0] != 100 or img.shape[1] != 100:
            img = cv2.resize(img, (100, 100))
        
        if img.dtype == np.uint8:
            img = img / 255.0
        
        return img.flatten().tolist()

    
    def query(self, query_img: Union[np.ndarray, Image.Image], limit: int = 1, with_distance: bool = True) -> list[dict]:
        query_field = 'PixelVec'
        
        query_value: list[float] = self.format_img(query_img)

        status, response = self.client.query(
            table_name=self.table_name,
            query_field=query_field,
            query_vector=query_value,
            limit=limit,
            with_distance=with_distance
        )

        print(response['message'])
        if status != 200:
            raise Exception(f'Query failed with status {status}')
        
        return response['result']