import os
import re
import sys
import time
from collections import deque
from configparser import ConfigParser

import discord
import torch
from PIL import Image

import LangModelAPI as m
import StableDiffusionAPI as sd
from CaptionerAPI import describe_image
from makeConfig import makeConfig

profile = sys.argv[1] if len(sys.argv) > 1 else 'default'

makeConfig(profile)

config = ConfigParser()
config.read('config.ini')

token = config[profile]['token']

help_text = """Commands:
    n - refers to the "n" most recent messages used for context in the channel.
    n2 - refers to the "n2" most recent messages to skip when using context in the channel.
    <> - required argument
    [] - optional argument
    ... - any number of arguments
    user_name - @mention of the user

    Text operations:
    `help` - display this message
    `summarize <n> [n2]` - summarize the last n messages, optionally skipping the last n2 messages
    `query <n> [n2] <...>` - query the chatbot with the given text, optionally skipping the last n2 messages. Ex: `query 10 2 What conclusions can we draw from this?`
    `response <...>` - respond to the chatbot with the given text
    `prompt <...>` - prompt the bare chatbot with the given text
    `roast <user_name> <n> [n2]` - roast the user with the given name using the context from the past n messages, optionally skipping the last n2 messages (Doesn't work well. Better prompt engineering needed)
    `act_like <user_name> <n> [n2]` - act like the user with the given name and respond as them. n is the number of messages for context, optionally skipping the last n2 messages

    Image operations:
    `generate [-waifu] <...>` - generate an image with the given prompt (normal Stable Diffusion) or waifu (hakurei/waifu-diffusion) if `-waifu` flag is set
    """
commands = [r'(?P<command>help)',
            r'(?P<command>summarize)\s+(?P<n>\d+)(?:\s+(?P<n2>\d*))?',
            r'(?P<command>query)\s+(?P<n>\d+)(?:\s+(?P<n2>\d+))?\s+(?P<text>.+)',
            r'(?P<command>response)\s+(?P<text>.+)',
            r'(?P<command>prompt)\s+(?P<text>.+)',
            r'(?P<command>roast)\s+(?P<user>[^\s]+)\s+(?P<n>\d+)(?:\s+(?P<n2>\d*))?',
            r'(?P<command>act_like)\s+(?P<user>[^\s]+)\s+(?P<n>\d+)(?:\s+(?P<n2>\d*))?',
            r'(?P<command>generate)\s+(?P<isWaifu>\-waifu)?\s*(?P<text>.+)']

class myClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user} ({self.user.mention})')
        self.terminal_size = os.get_terminal_size()[0]
        self.message_time_queue = deque(maxlen=12)
        self.conversation_history = {} # {channel_id: conversation}
        self.keep_history = False

    async def format_messages(self, content, message: discord.Message, n, n2='0'):
        if not n.isdigit():
            return
        n2 = int(n2) if isinstance(n2, str) and n2.isdigit() else 0
        sent_message_content = ''
        messages = message.channel.history(limit=int(n)+1)
        messages = [message async for message in messages][:n2:-1]
        sent_message = await message.channel.send('Working on it...', silent=True)
        self.message_time_queue.append(time.time())
        for message in messages:
            content = message.content
            for user in message.mentions:
                name = user.nick if isinstance(user, discord.Member) and user.nick else user.global_name
                name = name if name else user.name
                content = re.sub(f'<@!?{user.id}>', name, content)
            
            for embed in message.embeds:
                from pprint import pprint
                embed = embed.to_dict()
                if 'url' not in embed.keys():
                    continue
                url = embed['url']
                if 'thumbnail' in embed.keys():
                    url = embed['thumbnail']['url']
                
                if url.endswith('.png') or url.endswith('.jpg') or url.endswith('.jpeg'):
                    url_replacement = f"<{embed['type']}>{describe_image(url)}</{embed['type']}>"
                else:
                    url_replacement = ""
                
                content = re.sub(embed['url'], url_replacement, content)
                    

            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type == 'image/png' or attachment.content_type == 'image/jpeg':
                        content += f"<{attachment.content_type}>{describe_image(attachment.url)}</{attachment.content_type}>"


            name = message.author.nick if isinstance(message.author, discord.Member) and message.author.nick else message.author.global_name
            name = name if name else message.author.name
            if content:
                sent_message_content += f'<{name}>{content}</{name}>\n'
        return sent_message, sent_message_content

    async def send_message(self, generator, sent_message):
        t = time.time()
        self.terminal_size = os.get_terminal_size()[0]
        if sent_message.channel.id not in self.conversation_history or not self.keep_history:
            self.conversation_history[sent_message.channel.id] = next(generator)
        else:
            self.conversation_history[sent_message.channel.id] += next(generator).replace(self.conversation_history[sent_message.channel.id], '')
            self.keep_history = False
        for response in generator:
            if response and time.time() - t > 1.5 and len(self.message_time_queue) < self.message_time_queue.maxlen:
                t = time.time()
                await sent_message.edit(content=response)
                self.message_time_queue.append(time.time())
            while len(self.message_time_queue) and time.time() - self.message_time_queue[0] < 60:
                self.message_time_queue.popleft()
            print(response.split('\n')[-1][-self.terminal_size:], end='\r\r')
        print()
        self.conversation_history[sent_message.channel.id] += f'{response}<|endoftext|>'
        return await sent_message.edit(content=response)
    
    async def send_image(self, image: Image, sent_message):
        image.save('temp.jpg', quality=95, subsampling=0)
        image = discord.File('temp.jpg')
        await sent_message.channel.send(file=image, silent=True)

        # Delete the sent message
        await sent_message.delete()

    async def on_message(self, message):
        torch.cuda.empty_cache()
        if not message.content:
            return
        content = message.content.split()
        mentions = {user.id: user.name for user in message.mentions}
        if content.pop(0) != self.user.mention:
            return
        if not content:
            return
        content = ' '.join(content)

        mat = self.get_matching_command(content)

        if not mat:
            return

        command = mat.group('command')

        if command == 'help':
            await message.channel.send(help_text)
            return
        if command == 'summarize':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2'))
            return await self.send_message(m.summarize(sent_message_content), sent_message)
        if command == 'query':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2'))
            return await self.send_message(m.query(sent_message_content, mat.group('text')), sent_message)
        if command == 'response':
            sent_message = await message.channel.send('Working on it...')
            self.keep_history = True
            history = self.conversation_history[sent_message.channel.id] if sent_message.channel.id in self.conversation_history else None
            return await self.send_message(m.response(history, mat.group('text')), sent_message)
        if command == 'prompt':
            sent_message = await message.channel.send('Working on it...')
            return await self.send_message(m.prompt(mat.group('text')), sent_message)
        if command == 'roast':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2'))
            username = re.sub(r'<@!?(\d+)>', r'\1', mat.group('user'))
            username = mentions[int(username)] if username.isdigit() else username
            return await self.send_message(m.roast(sent_message_content, username), sent_message)
        if command == 'act_like':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2'))
            username = re.sub(r'<@!?(\d+)>', r'\1', mat.group('user'))
            username = mentions[int(username)] if username.isdigit() else username
            return await self.send_message(m.act_like(sent_message_content, username), sent_message)
        if command == 'generate':
            sent_message = await message.channel.send('Generating...')
            return await self.send_image(
                sd.generate(
                    mat.group('text'),
                    img_type='waifu' if mat.group('isWaifu') else 'normal',
                    width=512 if mat.group('isWaifu') else 1024,
                    height=512 if mat.group('isWaifu') else 1024,
                    num_inference_steps=120 if mat.group('isWaifu') else 40,
                    neg_prompt='lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry' if mat.group('isWaifu') else '',
                ),
                sent_message
            )

    def get_matching_command(self, content):
        for command in commands:
            if mat:=re.match(command, content):
                return mat
        return None

if __name__ == '__main__':
    client = myClient()
    client.run(token)
