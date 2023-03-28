import os
import sys
sys.path.append(os.getcwd())

from argparse import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import glob
import gzip as gz

import regex as re
from more_itertools import windowed
from datasets import Dataset
from typing import Dict

import logging
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

# ERRONEOUS_NEWLINES_RE = re.compile(r"\n(?!\u)")  # "\n" followed by any character that is not a unicode character
# ERRONEOUS_NEWLINES_RE = re.compile(ur"\n(?!\\u[0-9a-f])")  # "\n" followed by any character that is not a unicode character
def main(
    nn_name,
    token_chunk_len,
    min_word_per_chunk):

    # Locate data to be tokenized
    data_dir = os.path.join( './datasets/finetune/text_format/' )

    # Locate tokenizer to use
    tokenizer = AutoTokenizer.from_pretrained(nn_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Convert data to huggingface dataset    
    dataset = Dataset.from_generator( dataset_generator, gen_kwargs={'data_dir':data_dir} )

    dataset_dict = dataset.train_test_split( test_size=0.2, shuffle=True)

    # Breaking Large Texts into sub-texts Loop over dataset applying tokenizer to each 
    text_chunk_len = int( token_chunk_len * 0.5 ) # rough approximation of number of words per token
    
    dataset_dict = dataset_dict.map( lambda batch: break_up_text(batch, text_chunk_len, min_word_per_chunk), batched=False )
    dataset_dict = dataset_dict.map( lambda batch: {'text': batch['text'][0] }, batched=True  )

    # Loop over dataset applying tokenizer to each row
    dataset_dict = dataset_dict.map( lambda batch: map_tokenize(batch, tokenizer, max_len=token_chunk_len), batched=True  )

    # Save Dataset in torch format
    dataset_train = dataset_dict['train']
    dataset_val = dataset_dict['test']

    dataset_train.set_format(type='torch', columns=["text", "input_ids", "attention_mask"] )
    dataset_val.set_format(type='torch', columns=["text", "input_ids", "attention_mask"] )
    

    # Saving to disk
    dir_ = f'./data/finetune/preprocessed/{nn_name.replace("/","_")}'
    os.makedirs(dir_)

    dataset_train.save_to_disk( os.path.join(dir_,'train.arrow') )
    dataset_val.save_to_disk( os.path.join(dir_,'val.arrow') )

    logging.info('Finished Preprocessing Data')
    
    return None

def dataset_generator(data_dir:str):
    
    gen_fp = glob.glob( os.path.join(data_dir,'**','[0-9]'*3+'.txt' ) )

    for fp in gen_fp:

        # with gz.open(fp, "rb") as f:
            # text = f.read()
            # text = text.decode('utf-8')
        with open(fp, "r") as f:
            text = f.read()
            yield {'text':text}

def break_up_text(dict_text:Dict[str,str], max_words:int=200, min_word_per_chunk:int=10):
    # Split text into chunks of text with length M
    # One research paper, can now become i>1 input datums
    
    text = dict_text['text']

    # First split by paragraph
    li_text = split_paragraphs(text, min_word_per_chunk)

    # Splitting based on max_len
    li_li_text = [ list(map( lambda seq:  ' '.join(seq).strip(' '),
                        windowed( txt.split(' ') , max_words, step=int(max_words*3/4),fillvalue='' ) 
                        ))
                         for txt in li_text ] 

    li_text = sum(li_li_text, [])

    return {'text':li_text}


def split_paragraphs(input_text:str="", min_word_per_chunk:int=10):
    # Split text into paragraphs
    # Paragraphs are separated by two or more newlines
    # Paragraphs are returned with a trailing newline

    no_newlines = input_text.strip("\n")  # remove leading and trailing "\n"
    split_text = NEWLINES_RE.split(no_newlines)  # regex splitting

    # TODO: Ensure that new paragraph split is not made when colon (:) is at the end of a section \uf0b7 (bullet point)

    # remove '\n' markers within each text in split_text unless it is followed by \uf0b7(semicolon) or other \u markers
    # split_text = [ ERRONEOUS_NEWLINES_RE.sub(txt,'') for txt in split_text ]
    split_text = [ re.sub(  r"\n(?![^\u0000-\u007F]+)",'',txt) for txt in split_text]
    split_text = [ txt.strip(' ') for txt in split_text]

    # removing text chunks with low number of words
    split_text = [txt for txt in split_text if len(txt.split(' '))>= min_word_per_chunk ]

    # split_text = [p + "\n" for p in split_text if p.strip()]
    # p + "\n" ensures that all lines in the paragraph end with a newline
    # p.strip() == True if paragraph has other characters than whitespace

    
    return split_text


def map_tokenize(batch, tokenizer, max_len:int):
    # Tokenize each row of the dataset
    # batch['text'] is a list of strings
    
    outp = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_len)

    return outp



def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B' )
    parser.add_argument('--token_chunk_len',type=int, default=200 )
    parser.add_argument('--min_word_per_chunk',type=int, default=10)

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    main(**vars(args))

    



