import os
import re
import time
from collections import deque 

import discord
import torch

import OpenAssistant_Model as m

if os.path.exists('token.txt'):
    with open('token.txt', 'r') as f:
        token = f.readline().strip()
else:
    token = input('Enter your token:\n')
    cache_token = input('Cache token for future use? (y/n)\n').lower()
    if cache_token == 'y':
        with open('token.txt', 'w') as f:
            f.write(token)

help_text = """Commands:
    n - refers to the "n" most recent messages used for context in the channel.
    n2 - refers to the "n2" most recent messages to skip when using context in the channel.
    <> - required argument
    [] - optional argument
    ... - any number of arguments
    user_name - @mention of the user

    `help` - display this message
    `summarize <n> [n2]` - summarize the last n messages, optionally skipping the last n2 messages
    `query <n> [n2] <...>` - query the chatbot with the given text, optionally skipping the last n2 messages. Ex: `query 10 2 What conclusions can we draw from this?`
    `response <...>` - respond to the chatbot with the given text
    `prompt <...>` - prompt the bare chatbot with the given text
    `roast <user_name> <n> [n2]` - roast the user with the given name using the context from the past n messages, optionally skipping the last n2 messages (Doesn't work well. Better prompt engineering needed)
    `act_like <user_name> <n> [n2]` - act like the user with the given name and respond as them. n is the number of messages for context, optionally skipping the last n2 messages
    """
commands = [r'(?P<command>help)',
            r'(?P<command>summarize)\s+(?P<n>\d+)(?:\s+(?P<n2>\d*))?',
            r'(?P<command>query)\s+(?P<n>\d+)(?:\s+(?P<n2>\d+))?\s+(?P<text>.+)',
            r'(?P<command>response)\s+(?P<text>.+)',
            r'(?P<command>prompt)\s+(?P<text>.+)',
            r'(?P<command>roast)\s+(?P<user>[^\s]+)\s+(?P<n>\d+)(?:\s+(?P<n2>\d*))?',
            r'(?P<command>act_like)\s+(?P<user>[^\s]+)\s+(?P<n>\d+)(?:\s+(?P<n2>\d*))?',]

class myClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user} ({self.user.mention})')
        self.terminal_size = os.get_terminal_size()[0]
        self.message_time_queue = deque(maxlen=12)
        self.conversation_history = {} # {channel_id: conversation}

    async def format_messages(self, content, message, n, n2='0'):
        if not n.isdigit():
            return
        n2 = int(n2) if isinstance(n2, str) and n2.isdigit() else 0
        sent_message_content = ''
        messages = message.channel.history(limit=int(n)+1)
        messages = (await messages.flatten())[:n2:-1]
        sent_message = await message.channel.send('Working on it...')
        self.message_time_queue.append(time.time())
        for message in messages:
            content = message.content
            for user in message.mentions:
                content = re.sub(f'<@!?{user.id}>', user.name, content)
            
            sent_message_content += f'<|prompter|>{str(message.author)[:-5]}: {content}<|endoftext|>'
        return sent_message, sent_message_content

    async def send_message(self, generator, sent_message):
        t = time.time()
        if sent_message.channel.id not in self.conversation_history:
            self.conversation_history[sent_message.channel.id] = next(generator) + '<|endoftext|>'
        else:
            self.conversation_history[sent_message.channel.id] += next(generator) + '<|endoftext|>'
        for response in generator:
            if response and time.time() - t > 1.5 and len(self.message_time_queue) < self.message_time_queue.maxlen:
                t = time.time()
                await sent_message.edit(content=response)
                self.message_time_queue.append(time.time())
            while len(self.message_time_queue) and time.time() - self.message_time_queue[0] < 60:
                self.message_time_queue.popleft()
            print(response.split('\n')[-1][-self.terminal_size:], end='\r\r')
        self.conversation_history[sent_message.channel.id] += f'{response}<|endoftext|>'
        return await sent_message.edit(content=response)

    async def on_message(self, message):
        torch.cuda.empty_cache()
        if not message.content:
            return
        content = message.content.split()
        mentions = {user.id: user.name for user in  message.mentions}
        if content.pop(0) != self.user.mention:
            return
        if not content:
            return
        content = ' '.join(content)

        mat = self.get_matching_command(content)
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

    def get_matching_command(self, content):
        for command in commands:
            if mat:=re.match(command, content):
                return mat
        return None

if __name__ == '__main__':
    client = myClient()
    client.run(token, bot=False)
