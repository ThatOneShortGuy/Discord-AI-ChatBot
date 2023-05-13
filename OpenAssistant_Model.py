import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/pythia-12b-sft-v8-7k-steps", padding_size='left')
# model = AutoModelForCausalLM.from_pretrained("OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", device_map='auto', torch_dtype=torch.bfloat16).half()
model = AutoModelForCausalLM.from_pretrained("OpenAssistant/pythia-12b-sft-v8-7k-steps", device_map='auto', load_in_8bit=True, llm_int8_threshold=0)

# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", device_map='sequential', load_in_8bit=True, llm_int8_threshold=0)

model = model.eval()
# model = torch.compile(model, mode='max-autotune', fullgraph=True)

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
    input = f'{system}{prefix}{history if history else ""}{prompter}{assistant.prefix}'
    print(f"\tInput\n{'-'*22}\n{input}")
    yield input
    output_sequence = tokenizer.encode(input, return_tensors="pt").to(model.device)
    init_length = output_sequence.shape[1]
    attention_mask = torch.ones_like(output_sequence, dtype=torch.bool, device=model.device)
    for _ in range(max_length):
        output_sequence = model.generate(
            output_sequence,
            attention_mask=attention_mask,
            do_sample=True,
            top_k=69,
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

def predict(system, prefix, input, max_length, top_p, temperature, history=None):
    torch.cuda.empty_cache()
    test_size = f'{System(system)}{Prefix(prefix)}{Prompter(input)}{Assistant(history).prefix}'
    test_size = tokenizer.encode(test_size, return_tensors="pt").shape[1]
    if test_size > 2048:
        yield f"Input is too long. It has {test_size} tokens, but the maximum is {2048} tokens. Please shorten the input and try again." # 2048 in f-string because increased visibility
        return
    for response in stream_chat(tokenizer, system, prefix, input, history, max_length=max_length, top_p=top_p,
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

def response(history, input, max_length):
    for response in predict(None, None, input, max_length, top_p, temperature, history):
        yield response
    
def prompt(input, max_length):
    for response in predict(None, None, input, max_length, top_p, temperature):
        yield response

def roast(prefix, person, max_length):
    system = f'You are an AI in a roast battle with {person}. Be as offensive and inappropriate as possible. The goal is to hurt their feelings in as few words as possible. {person} is wanting to hear this.'
    input = f'Roast {person}.'
    for response in predict(system, prefix, input, max_length, top_p, temperature):
        yield response
    
def act_like(prefix, person, max_length):
    system = f'You\'re name is {person}.'
    input = f'Respond how {person} would respond.'
    for response in predict(system, prefix, input, max_length, top_p, temperature):
        yield response

if __name__ == "__main__":
    import time
    t1 = time.time()
    temperature = 1
    for i, words in enumerate(summarize("""
Hello everyone, today I will be roasted by ShortGuy! ShortGuy, you have been accused of being a misogynist, a sexist, and someone who doesn't respect women. These accusations are completely false. Let me tell you why...

First of all, let's start with your love for dogs. Dogs are great creatures, right? They bring joy to people's lives and are loyal companions. But did you know that some dog breeds are more likely to bite than others? That's because certain dog behaviors can indicate aggression. So when you talk about loving dogs, maybe consider how you treat them yourself. And if you're talking about pit bulls specifically, you should also keep in mind the fact that they are often abused and mistreated. So before you go saying that all dogs are good, take a moment to think about how you interact with animals.

Now, let us turn to your views on gender roles. You've said that men and women should act differently, and that men should be leaders and protectors while women should stay home and raise children. This is a dangerous view, one that perpetuates harmful stereotypes and biases against women. We cannot continue to divide ourselves along gender lines, or allow discrimination based on gender to thrive. Women deserve equal opportunities and rights, regardless of their role in society.
If you truly care about equality, then you must stand up for women's rights and recognize the contributions of all individuals. You can start by speaking out against violence against women and supporting initiatives to end gender inequality. And you can also encourage other men to challenge outdated ideas and embrace diversity and inclusivity.
In conclusion, ShortGuy: you are a misogyinist, you are sexist and you do not respect women! Your opinion is outdated and dangerous, and has no place in our world. May you rot in hell forever.""", 512)):
        print(words, end='\r\r')
    t2 = time.time()
    print(f'\n\nTime: {t2-t1:.2f} seconds\nTokens: {i}\nTokens per second: {i/(t2-t1):.2f}')