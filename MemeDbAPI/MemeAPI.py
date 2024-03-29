# https://epsilla-inc.gitbook.io/epsilladb/api-reference      API Reference

from typing import Union
from pyepsilla import vectordb
import numpy as np
import cv2
from PIL import Image

class Database:
    def __init__(self, table_name: str, host: str = '192.168.1.16', port: str = '8888'):
        self.client = vectordb.Client(host=host, port=port)
        self.client.load_db(db_name='MyDB', db_path='/tmp/epsilla')
        self.client.use_db('MyDB')
        self.table_name = table_name
    
    def insert(self, records: list[dict]):
        return self.client.insert(
            table_name=self.table_name,
            records=records
        )
    
    def format_img(self, img: Union[np.ndarray, Image.Image]) -> list[float]:
        if isinstance(query_img, Image.Image):
            query_img = np.array(query_img)
            query_img = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
        
        if query_img.shape[0] != 100 or query_img.shape[1] != 100:
            query_img = cv2.resize(query_img, (100, 100))
        
        if query_img.dtype == np.uint8:
            query_img = query_img / 255.0
        
        return query_img.flatten().tolist()

    
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