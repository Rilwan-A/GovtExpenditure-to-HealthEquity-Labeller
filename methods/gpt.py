import os
import openai
import yaml
from argparse import ArgumentParser
from typing import List
import logging
from tqdm import tqdm
api_keys = yaml.safe_load(open( os.path.join('methods','api_keys.yaml'), 'r'))

openai.api_key = api_keys["openai"]

import time

rate_limit_per_minute = 25 #60
delay = 60.0 / rate_limit_per_minute

token_limit_per_minute = 150000


class OpenAICompletion():

    def __init__(self, openai_model, max_tokens, temperature, logprobs):
        self.openai_model = openai_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logprobs = logprobs
    
    def get_completions_batched( self, li_prompts:List[str]  ) -> List[str]:

        li_responses = []

        batch_size = min( int(token_limit_per_minute / (self.max_tokens * 30) ), 20 )

        for idx in tqdm(range(0, len(li_prompts), batch_size )):
            
            batch_responses = self.get_completions( li_prompts[idx:idx+batch_size] )
            li_responses.extend( batch_responses )

        return li_responses 


    def get_completions( self, li_prompts:List[str]  ) -> List[str]:
        global delay

        suceeded = False
        time.sleep(delay)
        while suceeded == False:

            try:
                response = openai.Completion.create(
                    model = self.openai_model,
                    prompt = li_prompts,
                    max_tokens = self.max_tokens,
                    temperature = self.temperature
                )
                suceeded = True

            except Exception as e:
                print(e)
                time.sleep(20)
                
                delay = delay*1.25
                
                           
            

        li_completions = [None]*len(li_prompts)
        for choice in response.choices:
            li_completions[choice.index] = choice.text
        
        return li_completions

    
    def parse_args( parent_parser=None ):

        if parent_parser is not None:
            parser = ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False,)
        else:
            parser = ArgumentParser()

        
        parser.add_argument('--openai_model', type=str, default='text-davinci-002',
            choices=['text-ada-001','text-babbage-001','text-curie-001','text-davinci-003',
            'text-davinci-002', 'curie-instruct-beta','davinci-instruct-beta' ] )

        parser.add_argument('--max_tokens', type=int, default=3 )
        parser.add_argument('--temperature', type=int, default=0.7 ) #Use low temperature when straight forward answers https://platform.openai.com/docs/models/finding-the-right-model
        parser.add_argument('--logprobs', type=bool, default=None)

        
        args = parser.parse_known_args()[0]

        if any( (str_ in args.openai_model for str_ in  ['ada','babbage','curie'] ) ):
            print('When using a weaker model such as ada or babbage, use a very low temprature to ensure a definitive yes/no answer is retreived')
        return args