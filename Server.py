import asyncio
import os
import re
import sys
import time
from collections import deque
from configparser import ConfigParser
from typing import Union

import discord
import requests
import torch
from PIL import Image

import LangModelAPI as LangModel
import StableDiffusionAPI as sd
from CaptionerAPI import describe_image
from makeConfig import makeConfig
from CacheUsers import UserCache
from MemeDbAPI import MemeDatabase, get_image_embedding

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
        print(f'Logged on as {self.user} ({self.user.mention})') # type: ignore
        self.model_info = LangModel.get_model_info()
        self.terminal_size = os.get_terminal_size()[0]
        self.message_time_queue: deque[float] = deque(maxlen=24)
        self.conversation_history: dict[int, str] = {} # {channel_id: conversation}
        self.keep_history = False
        self.user_cache = UserCache()
        self.meme_client = MemeDatabase('UCASEmbeddings', config[profile]['MemeDB_ip'], config[profile]['MemeDB_port'], processing_function=get_image_embedding)
    
    def get_user_name(self, user: Union[discord.User, discord.Member]):
        name = self.user_cache.get_user(str(user.id))
        if not name:
            name = user.nick if isinstance(user, discord.Member) and user.nick else user.global_name
            name = name if name else user.name
            self.user_cache.add_user(str(user.id), name) if not user.bot else self.user_cache.add_bot(name)

        if user.bot:
            name = self.user_cache.get_user('bots', name)
        return name
    
    async def edit_message(self, message: discord.Message, content: str, no_check=False):
        if len(self.message_time_queue) and time.time() - self.message_time_queue[0] > 60 or no_check:
            self.message_time_queue.popleft()
        if len(self.message_time_queue) < self.message_time_queue.maxlen and (not len(self.message_time_queue) or (time.time() - self.message_time_queue[-1]) > 1.5) or no_check: # type: ignore
            self.message_time_queue.append(time.time())
            return await message.edit(content=content)

    async def format_messages(self, content, message: discord.Message, n, n2='0'):
        if not n.isdigit():
            return
        n2 = int(n2) if isinstance(n2, str) and n2.isdigit() else 0
        sent_message_content = ''
        messages = message.channel.history(limit=int(n)+1)
        messages = [message async for message in messages][:n2:-1]
        sent_message = await message.channel.send('Working on it...', silent=True)
        self.message_time_queue.append(time.time())
        previous_author = None
        num_messages = len(messages)
        for message_num, message in enumerate(messages):
            content = message.content
            for user in message.mentions:
                name = self.get_user_name(user)
                content = re.sub(f'<@!?{user.id}>', name, content)
            
            for embed in message.embeds:
                url = embed.url
                if not url:
                    continue
                url = re.match(r'^(.+\.((png)|(jpg)|(jpeg))).*', url)
                if not url:
                    continue
                url = url.group(1)
                
                if url.endswith('.png') or url.endswith('.jpg') or url.endswith('.jpeg'):
                    await self.edit_message(sent_message, f'Describing {url}')
                    url_replacement = f"<{embed['type']}>{describe_image(url)}</{embed['type']}>" # type: ignore
                else:
                    url_replacement = ""
                
                embed = embed.to_dict()
                embed['url'] = re.sub(r'([(^)|*$])', r'\\\1', embed['url']) # type: ignore
                content = re.sub(embed['url'], url_replacement, content)
                    

            if message.attachments:
                for attachment in message.attachments:
                    if attachment.content_type == 'image/png' or attachment.content_type == 'image/jpeg':
                        await sent_message.edit(content=f'Analyzing {attachment.url}\n{message_num+1}/{num_messages} ({(message_num+1)/num_messages:.2%}) messages')
                        content += f"<{attachment.content_type}>{describe_image(attachment.url)}</{attachment.content_type}>"


            name = self.get_user_name(message.author)
            if content and previous_author != name:
                if previous_author:
                    sent_message_content += self.model_info["end_token"] + '\n'
                sent_message_content += f'{self.model_info["start_token"]}{name}\n{content}'
                previous_author = name
            elif content:
                sent_message_content += f'\n{content}'
            
        sent_message_content += self.model_info["end_token"] + '\n'
        sent_message_content = re.sub(r'<\/(.*?)>\s+<\1>', r'\n', sent_message_content)
        return sent_message, sent_message_content

    async def send_message(self, generator, sent_message: discord.Message):
        self.terminal_size = os.get_terminal_size()[0]
        if sent_message.channel.id not in self.conversation_history or not self.keep_history:
            self.conversation_history[sent_message.channel.id] = next(generator)
        else:
            self.conversation_history[sent_message.channel.id] += next(generator).replace(self.conversation_history[sent_message.channel.id], '')
            self.keep_history = False
        response = ''
        for response in generator:
            if re.match(r'^\s*$', response):
                await self.edit_message(sent_message, '[Empty response]', no_check=True)
                continue
            await self.edit_message(sent_message, response)
            print(response.split('\n')[-1][-self.terminal_size:], end='\r\r')
        print()
        self.conversation_history[sent_message.channel.id] += f'{response}{self.model_info["end_token"]}\n'
        return await self.edit_message(sent_message, response, no_check=True)
    
    async def send_image(self, image: Image.Image, sent_message):
        image.save('temp.jpg', quality=95, subsampling=0)
        discord_image = discord.File('temp.jpg')
        await sent_message.channel.send(file=discord_image, silent=True)

        # Delete the sent message
        await sent_message.delete()
    
    async def reply_and_react(self, message: discord.Message, reaction: str, response: str):
        try:
            await message.add_reaction(reaction)
        except discord.errors.Forbidden:
            response = f'{reaction} {response}'
        await message.reply(response, mention_author=False)

    async def on_meme(self, message: discord.Message):
        imgs_to_add_to_db: list[Image.Image] = []
        await asyncio.sleep(2)
        messages = [m async for m in message.channel.history(limit=50) if m.id == message.id][0]
        for embedObj in messages.embeds:
            url = embedObj.url
            if not url:
                continue
            url = re.match(r'^(.+\.((png)|(jpg)|(jpeg))).*', url)
            if not url:
                continue
            url = url.group(1)
            imgs_to_add_to_db.append(Image.open(requests.get(url, stream=True).raw))
            
        for attachment in messages.attachments:
            if attachment.content_type and  attachment.content_type.startswith('image'):
                imgs_to_add_to_db.append(Image.open(requests.get(attachment.url, stream=True).raw))

        if not imgs_to_add_to_db:
            return
        
        img_vec = []
        for image in imgs_to_add_to_db:
            response = self.meme_client.query(image)[0]
            confidence = 1 - response['@distance']
            if confidence < .69:
                print(f'Image not in database with distance {response["@distance"]} ({confidence:.2%} confidence)') # type: ignore
                img_vec.append(self.meme_client.format_img(image))
                continue
            print(f'Image already in database with distance {response["@distance"]}') # type: ignore
            discord_link = f'https://discord.com/channels/{message.guild.id}/{message.channel.id}/{response["MessageID"]}' # type: ignore
            await self.reply_and_react(message, '♻️', f'Meme already posted {discord_link} with {confidence:.2%} confidence') # type: ignore

        if not img_vec:
            return
        
        status, response = self.meme_client.insert([{'MessageID': str(messages.id), 'PixelVec': img} for img in img_vec])
        print(response['message'])

    async def on_message(self, message: discord.Message):
        if message.channel.id == int(config[profile]['meme_channel_id']):
            await self.on_meme(message)
        if not message.content:
            return
        content = message.content.split()
        if content.pop(0) != self.user.mention: # type: ignore
            return
        if not content:
            return
        mentions = {user.id: self.get_user_name(user) for user in message.mentions}
        content = ' '.join(content)

        mat = self.get_matching_command(content)

        if not mat:
            return

        command = mat.group('command')

        if command == 'help':
            await message.channel.send(help_text)
            return
            
        if command == 'summarize':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2')) # type: ignore
            return await self.send_message(LangModel.summarize(sent_message_content), sent_message)
            
        if command == 'query':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2')) # type: ignore
            return await self.send_message(LangModel.query(sent_message_content, mat.group('text')), sent_message)
            
        if command == 'response':
            sent_message = await message.channel.send('Working on it...')
            self.keep_history = True
            history = self.conversation_history[sent_message.channel.id] if sent_message.channel.id in self.conversation_history else None
            return await self.send_message(LangModel.response(history, mat.group('text')), sent_message)
            
        if command == 'prompt':
            sent_message = await message.channel.send('Working on it...')
            return await self.send_message(LangModel.prompt(mat.group('text')), sent_message)
            
        if command == 'roast':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2')) # type: ignore
            username = re.sub(r'<@!?(\d+)>', r'\1', mat.group('user'))
            username = mentions[int(username)] if username.isdigit() else username
            return await self.send_message(LangModel.roast(sent_message_content, username), sent_message)
            
        if command == 'act_like':
            sent_message, sent_message_content = await self.format_messages(content, message, mat.group('n'), mat.group('n2')) # type: ignore
            username = re.sub(r'<@!?(\d+)>', r'\1', mat.group('user'))
            username = mentions[int(username)] if username.isdigit() else username
            return await self.send_message(LangModel.act_like(sent_message_content, username), sent_message)
            
        if command == 'generate':
            sent_message = await message.channel.send('Generating...')
            image = sd.generate(
                mat.group('text'),
                img_type='waifu' if mat.group('isWaifu') else 'normal',
                width=512 if mat.group('isWaifu') else 512,
                height=512 if mat.group('isWaifu') else 512,
                num_inference_steps=120 if mat.group('isWaifu') else 40,
                neg_prompt='lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry' if mat.group('isWaifu') else '',
            ),
            if isinstance(image, Image.Image):
                return await self.send_image(
                    image,
                    sent_message
                )
            return await self.edit_message(sent_message, image) # type: ignore

    def get_matching_command(self, content):
        for command in commands:
            if mat:=re.match(command, content):
                return mat
        return None

if __name__ == '__main__':
    client = myClient()
    client.run(token)
