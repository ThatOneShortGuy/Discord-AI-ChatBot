# Introduction
This is a simple project that uses open-source models from [HuggingFace](https://huggingface.co/) to integrate with Discord.

# Requirements
- Python 3.8+
- PyTorch + CUDA

# Setup
1. Ensure the [requirements](#Requirements) are met
2. Clone the repository
3. Install the requirements
- [Pytorch](https://pytorch.org/get-started/locally/)
```bash
python3 -m pip install -r requirements.txt
```
4. Create a Discord bot and get the token. Keep this ready for later.
5. Run `makeConfig.py` to create the config file. It will ask you for the token.
```bash
python3 makeConfig.py
```
**OR** The when you run the bot for the first time, it will ask you for the token.

It will ask if you want to use the default settings. If you are running all the servers locally, on the same computer, then you can use the default settings. Otherwise, you will need to change the settings in the config file. The settings are explained [below](#default-config).

# Usage
1. Run the bot
```bash
python3 Server.py
```
2. `@mention` the bot and type `help` to get a list of commands. (e.g. `@bot help`)

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

## Config Generation
The config file is generated using [makeConfig.py](makeConfig.py). It will ask you for the Discord bot token and if you want to use the default settings. If you are running all the servers locally, on the same computer, then you can use the default settings. Otherwise, you will need to change the settings in the config file. The settings are explained [below](#default-config).

### Default Config
- token - Discord bot token
- cuda_visible_devices - CUDA devices to use (e.g. 0,1,2,3) (default: 0)
- language_model - The language model to use (default: OpenAssistant/falcon-7b-sft-mix-2000)
- image_gen_model - The image generation model to use (default: stabilityai/stable-diffusion-2-1)
- image_description_ip - The IP address of the image description server (default: 127.0.0.1)
- image_description_port - The port of the image description server (default: 5002)
- model_server_ip - The IP address of the model server (default: 127.0.0.1)
- model_server_port - The port of the model server (default: 5000)
- image_gen_ip - The IP address of the image generation server (default: 127.0.0.1)
- image_gen_port - The port of the image generation server (default: 5001)
- transformers_cache - The cache directory for transformers (default: "" (transformers default cache directory))

### Generating specific profiles
To generate a new profile, run [makeConfig.py](makeConfig.py) with an extra argument for the profile name. (e.g. `python3 makeConfig.py profile1`)

### Using a specific profile
To use a specific profile, run [Server.py](Server.py) with an extra argument for the profile name. (e.g. `python3 Server.py profile1`)
Or, each server can be run individually with the profile name as an argument. (e.g. `python3 HostInferenceServer.py profile1`)

## Image generation
The image generation uses [Stable Diffusion 2](https://huggingface.co/stabilityai/stable-diffusion-2-1) to generate images. It also communicates to a server found in
[stableInferenceServer.py](stableInferenceServer.py) to generate images. The server is a Flask server that uses the model to generate images.