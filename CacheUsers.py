import json
from typing import Optional

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
    
    def add_bot(self, nick: str) -> None:
        if 'bots' not in self.cache.keys():
            self.cache['bots'] = {}
        if nick not in self.cache['bots'].keys():
            self.cache['bots'][nick] = ""
            self.save_cache()
    
    def get_user(self, userid: str, nick: Optional[str] = None) -> str:
        self.load_cache()
        if userid == 'bots':
            if not nick:
                raise ValueError('Bot nick not specified.')
            self.add_bot(nick)
            return self.cache['bots'][nick] if self.cache['bots'][nick] else nick
        if userid in self.cache:
            user = self.cache[userid]
            if user['name']:
                return user['name']
        return ''
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)
    