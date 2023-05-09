import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")
model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", device_map='auto', torch_dtype=torch.bfloat16).half()

# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", device_map='sequential', torch_dtype=torch.bfloat16).half()

model = model.eval()

max_length = 256
top_p = 0.7
temperature = 0.95

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

def stream_chat(tokenizer, system_input, prefix_input, input, history, max_length, top_p, temperature):
    system = System(system_input)
    prefix = Prefix(prefix_input)
    prompter = Prompter(input)
    assistant = Assistant(history)
    input = f'{system}{prefix}{prompter}{assistant.prefix}'
    print(f"\tInput\n{'-'*20}\n{input}")
    output_sequence = tokenizer.encode(input, return_tensors="pt").to(model.device)
    init_length = output_sequence.shape[1]
    attention_mask = torch.ones_like(output_sequence, dtype=torch.bool)

    for _ in range(max_length):
        output_sequence = model.generate(
            output_sequence,
            attention_mask=attention_mask,
            do_sample=True,
            top_k=50,
            top_p=top_p,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            temperature=temperature,
            use_cache=True,
            no_repeat_ngram_size=3,
            early_stopping=True,
            max_new_tokens=1,)
        attention_mask = torch.ones_like(output_sequence, dtype=torch.bool)
        if tokenizer.decode(output_sequence[0][-1].tolist()) == tokenizer.eos_token:
            break
        yield tokenizer.decode(output_sequence[0][init_length:].tolist())

model.stream_chat = stream_chat

def predict(system, prefix, input, max_length, top_p, temperature, history=None):
    torch.cuda.empty_cache()
    test_size = f'{System(system)}{Prefix(prefix)}{Prompter(input)}{Assistant(history).prefix}'
    test_size = tokenizer.encode(test_size, return_tensors="pt").shape[1]
    if test_size > 2048:
        yield f"Input is too long. It has {test_size} tokens, but the maximum is {2048} tokens. Please shorten the input and try again." # 2048 in f-string because increased visibility
        return
    for response in model.stream_chat(tokenizer, system, prefix, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        yield response

def summarize(prefix: str, max_length):
    system = 'You are an AI that summarizes a conversation in as few words as possible.'
    input = f'Summarize the previous conversation in as few words as possible.'
    for response in predict(system, prefix, input, max_length, top_p, temperature):
        yield response
    
def query(conversation, query, max_length):
    system = "You are an AI named ChatGLM that answers questions based on a conversation in as few words as possible."
    for response in predict(system, conversation, query, max_length, top_p, temperature):
        yield response
    
def prompt(input, max_length):
    for response in predict(None, None, input, max_length, top_p, temperature):
        yield response
    
def act_like(prefix, person, max_length):
    system = f'You\'re name is {person}.'
    input = f'Respond how {person} would respond.'
    for response in predict(system, prefix, input, max_length, top_p, temperature):
        yield response

if __name__ == "__main__":
    for words in summarize("""
ShortGuy#3808: summarize the 5 articles
YourMom#3345: Working on it...
ShortGuy#3808: ShortGuy summarize in 
ShortGuy#3808: ShortGuy#3808: ShortGuy summarize 5
ShortGuy#3808: Working on it...
ShortGuy#3808: ShortGuy summarize 3""", 512):
        print(words, end='\r\r')