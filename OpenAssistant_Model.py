import json
import os
import socket
import time

import requests

os.system("")

SERVER_IP = (([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0]
URL = f'http://{SERVER_IP}:5000/generate_stream'

class System:
    def __init__(self, content, prefix='<|system|>', end_token='<|endoftext|>'):
        self.content = content
        self.prefix = prefix
        self.end_token = end_token
    
    def __str__(self):
        return f'{self.prefix}{self.content}{self.end_token}' if self.content else ''
    
    def __repr__(self):
        return self.__str__()
    
class Prompter(System):
    def __init__(self, content, prefix='<|prompter|>', end_token='<|endoftext|>'):
        super().__init__(content, prefix, end_token)

class Assistant(System):
    def __init__(self, content, prefix='<|assistant|>', end_token='<|endoftext|>'):
        super().__init__(content, prefix, end_token)

class Prefix(System):
    def __init__(self, content, prefix='<|prefix_begin|>', end_token='<|prefix_end|>'):
        super().__init__(content, prefix, end_token)

def stream_chat(system_input, prefix_input, input, history=None, custom_input=None):
    system = System(system_input)
    prefix = Prefix(prefix_input)
    prompter = Prompter(input)
    assistant = Assistant(history)
    input = f'{system}{prefix}{history if history else ""}{prompter}{assistant.prefix}' if custom_input is None else custom_input
    terminal_size = os.get_terminal_size()[0]
    print(f"{'Input'.center(terminal_size)}\n{'-'*terminal_size}\n\033[92m{input}\033[0m")
    yield input
    data = json.dumps({'text': input})
    headers = {'content-type': 'application/json'}
    # Stream the generated output
    err = False
    while True:
        try:
            response = requests.post(URL, data=data, headers=headers, stream=True)
        except requests.exceptions.ConnectionError:
            if err:
                time.sleep(3)
                continue
            yield f"Failed to connect to server. Attemping to start server. Please wait..."
            if os.name == 'nt':
                os.system("start python HostInferenceServer.py")
            else:
                os.system("python3 HostInferenceServer.py")
            time.sleep(5)
            yield f"Failed to connect to server. Attemping to start server. Please wait...\nServer starting. Please wait..."
            err = True
            continue
        break 
        
    for content in response.iter_content(chunk_size=None, decode_unicode=True):
        yield content
    

def predict(system, prefix, input, history=None):
    for response in stream_chat(system, prefix, input, history):
        yield response

def summarize(prefix: str):
    system = 'Summarize the conversation'
    input = f'How many different topics are there? What are they?'
    for response in stream_chat(system, prefix, input):
        yield response
    
def query(conversation, query):
    system = "- You are an AI named ChatGLM\n- Answer questions from a conversation\n- Answer in as few words as possible"
    for response in stream_chat(system, conversation, query):
        yield response

def response(history, input):
    for response in stream_chat(None, None, input, history):
        yield response
    
def prompt(input):
    system = 'Respond in as few words as possible.'
    for response in stream_chat(system, None, input):
        yield response

def roast(prefix, person):
    system = f'''You are in a roast battle with {person}. Roasting is:
- A verbal fight
- Insulting someone
- Hurt {person}'s feelings
- Just a joke'''
    input = f'Offend {person} in minimal words'
    system = System(system)
    prefix = Prefix(prefix)
    input = Prompter(input)
    # for response in stream_chat(system, prefix, input):
    for response in stream_chat(None, None, None, custom_input=f'{system}{prefix}{input}{Assistant(None).prefix}{person} is'):
        yield f'{person} is' + response
    
def act_like(prefix, person):
    for response in stream_chat(None, None, None, custom_input=f'{prefix}<|prompter|>{person}: '):
        yield response

if __name__ == "__main__":
    import time
    t1 = time.time()
    for i, words in enumerate(summarize("""
RoastedByAI: Hello everyone, today I will be roasting ShortGuy! ShortGuy, you have been accused of being a misogynist, a sexist, and someone who doesn't respect women. These accusations are completely false. Let me tell you why...

First of all, let's start with your love for dogs. Dogs are great creatures, right? They bring joy to people's lives and are loyal companions. But did you know that some dog breeds are more likely to bite than others? That's because certain dog behaviors can indicate aggression. So when you talk about loving dogs, maybe consider how you treat them yourself. And if you're talking about pit bulls specifically, you should also keep in mind the fact that they are often abused and mistreated. So before you go saying that all dogs are good, take a moment to think about how you interact with animals.

Now, let us turn to your views on gender roles. You've said that men and women should act differently, and that men should be leaders and protectors while women should stay home and raise children. This is a dangerous view, one that perpetuates harmful stereotypes and biases against women. We cannot continue to divide ourselves along gender lines, or allow discrimination based on gender to thrive. Women deserve equal opportunities and rights, regardless of their role in society.
If you truly care about equality, then you must stand up for women's rights and recognize the contributions of all individuals. You can start by speaking out against violence against women and supporting initiatives to end gender inequality. And you can also encourage other men to challenge outdated ideas and embrace diversity and inclusivity.
In conclusion, ShortGuy, you are a misogyinist, you are sexist and you do not respect women! Your opinion is outdated and dangerous, and has no place in our world. May you rot in hell forever.""")):
        print(words, end='\r\r')
    t2 = time.time()
    print(f'\n\nTime: {t2-t1:.2f} seconds\nTokens: {i}\nTokens per second: {i/(t2-t1):.2f}')