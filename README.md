# Introduction
This is a simple project that uses open-source models from [HuggingFace](https://huggingface.co/) to integrate with Discord.

# Installation
1. Clone the repository
2. Install the requirements
```bash
python3 -m pip install -r requirements.txt
```
3. Create a Discord bot and get the token. The when you run the bot for the first time, it will ask you for the token. It will then ask if you want it cached in a file. If you say yes, it will be cached in a file called `token.txt`.

# Usage
1. Run the bot
```bash
python3 Server.py
```
2. @mention the bot and type `help` to get a list of commands. (e.g. `@bot help`)

# Commands
## Syntax
- n - refers to the "n" most recent messages used for context in the channel. (int)
- n2 - refers to the "n2" most recent messages to skip when using context in the channel. (int)
- <> - required argument
- [] - optional argument
- ... - any number of arguments
- user_name - @mention of the user

## Text operations
- `help` - display this message
- `summarize <n> [n2]` - summarize the last n messages, optionally skipping the last n2 messages
- `query <n> [n2] <...>` - query the chatbot with the given text, optionally skipping the last n2 messages. (e.g. `query 10 2 What conclusions can we draw from this?`)
- `response <...>` - respond to the chatbot with the given text
- `prompt <...>` - prompt the bare chatbot with the given text
- `roast <user_name> <n> [n2]` - roast the user with the given name using the context from the past n messages, optionally skipping the last n2 messages (Doesn't work well. Better prompt engineering needed)
- `act_like <user_name> <n> [n2]` - act like the user with the given name and respond as them. n is the number of messages for context, optionally skipping the last n2 messages

## Image operations
- `generate <...>` - generate an image with the given prompt (normal Stable Diffusion)

# Details
## Chatbot
The main chatbot uses the [Open Assistant Falcon 7B SFT model](https://huggingface.co/OpenAssistant/falcon-7b-sft-mix-2000).
All prompts can be found in [OpenAssistantModel.py](OpenAssistantModel.py). It starts the server found in [HostInferenceServer.py](HostInferenceServer.py) and sends requests to it
if it's not already running. The server is a Flask server that uses the [Open Assistant Falcon 7B SFT model](https://huggingface.co/OpenAssistant/falcon-7b-sft-mix-2000) to generate responses.

## Image generation
The image generation uses [Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2-1) to generate images. It also communicates to a server found in
[stableInferenceServer.py](stableInferenceServer.py) to generate images. The server is a Flask server that uses the model to generate images.