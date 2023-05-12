import os
import openai
import yaml
from argparse import ArgumentParser
from typing import List
import logging
from tqdm import tqdm

if 'OPENAI_API_KEY' in os.environ:
    openai.api_key = os.environ["OPENAI_API_KEY"]

import time

requests_limit_per_minute = 20 #60
requests_limit_per_minute_adj = 19 #60
token_limit_per_minute = 150000


class OpenAICompletion():

    def __init__(self, openai_model, max_tokens, temperature, system_start=None):
        self.openai_model = openai_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_start = system_start if system_start else "Assistant is a large language model trained by OpenAI."
    
    
    def get_responses( self, li_prompts:List[str]  ) -> List[str]:

        li_responses = []

        # batch_size = min( int(token_limit_per_minute / (self.max_tokens * 30) ), 20 )
        
        start_time = time.time()

        for idx in tqdm(range(0, len(li_prompts) )):
            
            if idx % 150 == 0:
                time.sleep( 20 )
            else:
                time.sleep(  2.3 )
            
            batch_responses = self.get_completion( li_prompts[idx:idx+1] )
            
            li_responses.extend( batch_responses )

        return li_responses 


    def get_completion( self, li_prompts:List[str]  ) -> List[str]:
        
        assert len(li_prompts) == 1, "Only one prompt is supported at a time."

        suceeded = False
        while suceeded == False:

            try:
                response = openai.ChatCompletion.create(
                    model = self.openai_model,
                    messages = [{'role':'system', 'content':self.system_start} ] + \
                                 [ {'role':'user', 'content': prompt } for prompt in li_prompts] ,
                    max_tokens = self.max_tokens,
                    temperature = self.temperature
                )
                suceeded = True

            except Exception as e:
                print(e)
                time.sleep(60)
                           
        li_completions = ['NA']*len(li_prompts)
        for choice in response.choices:
            li_completions[choice['index']] = choice['message']['content']
        
        return li_completions

    @staticmethod
    def parse_args( parent_parser=None ):

        if parent_parser is not None:
            parser = ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False,)
        else:
            parser = ArgumentParser()

        
        parser.add_argument('--openai_model', type=str, default='gpt-3.5-turbo' )

        parser.add_argument('--max_tokens', type=int, default=3 )
        parser.add_argument('--temperature', type=int, default=0.7 ) #Use low temperature when straight forward answers https://platform.openai.com/docs/models/finding-the-right-model
                
        args = parser.parse_known_args()[0]

        if any( (str_ in args.openai_model for str_ in  ['ada','babbage','curie'] ) ):
            print('When using a weaker model such as ada or babbage, use a very low temprature to ensure a definitive yes/no answer is retreived')
        return args