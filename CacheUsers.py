import json
from pprint import pprint

class UserCache:
    def __init__(self, cache_file='usercache.json'):
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()
    
    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            print('No user cache found. Creating new cache file.')
    
    def add_user(self, userid: str, nick: str) -> None:
        if userid not in self.cache.keys():
            self.cache[userid] = {'nick': nick, 'name': ''}
            self.save_cache()
    
    def get_user(self, userid: str) -> str:
        if userid in self.cache:
            user = self.cache[userid]
            if user['name']:
                return user['name']
        return ''
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)
    