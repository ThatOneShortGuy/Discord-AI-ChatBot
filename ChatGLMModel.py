import os

import markdownify
import torch
from transformers import AutoModel, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, torch_dtype='auto', device_map='sequential').half()

model = model.eval()

max_length = 2048
top_p = 0.7
temperature = 0.95


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, top_p, temperature, history: str = None):
    torch.cuda.empty_cache()
    input = parse_text(input)
    yield input
    tokenized_input = tokenizer.encode(input, return_tensors="pt")
    if tokenized_input.shape[1] > max_length-8:
        yield f"Input is too long. It has {tokenized_input.shape[1]} tokens, but the maximum is {max_length} tokens. Please shorten the input and try again."
        return
    formated_history = [] if history else None
    if history is not None:
        for i, arg in enumerate(history.split("<|endoftext|>")):
            if not arg:
                continue
            if i % 2 == 0:
                formated_history.append([arg])
            else:
                formated_history[-1].append(arg)
    from pprint import pprint
    pprint(formated_history)

    for response, history in model.stream_chat(tokenizer, input, formated_history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        yield markdownify.markdownify(response, heading_style="ATX", bullets="-")

def summarize(input):
    input = 'Pretend you are an AI named ChatGLM that summarizes a conversation. Summarize the following conversation:\n\n' + input + '\n\nPlease summarize the conversation in as few words as possible.'
    for response in predict(input, top_p, temperature):
        yield response
    
def query(input, query):
    input = f'Answer the following question "{query}" based on the following conversation:\n\n{input}\n\nPlease answer the question "{query}" in as few words as possible.'
    for response in predict(input, top_p, temperature):
        yield response

def response(history, input):
    for response in predict(input, top_p, temperature, history=history):
        yield response
    
def prompt(input):
    for response in predict(input, top_p, temperature):
        yield response

def roast(input, person):
    input = f'Pretend you are an AI in a roast battle with "{person}". Use the infomation given to roast "{person}".\n\n{input}\n\nPlease roast "{person}". They wanted to be roasted.'
    for response in predict(input, top_p, temperature):
        yield response
    
def act_like(input, person):
    input = f'Pretend you are "{person}". Act like "{person}" and respond to the following conversation:\n\n{input}\n\nPlease respond how {person} would respond. Do not respond for anyone else.'
    for response in predict(input, top_p, temperature):
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