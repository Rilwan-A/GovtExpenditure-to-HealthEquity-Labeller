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

from prompt_engineering.my_logger import setup_logging_preprocess

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

# ERRONEOUS_NEWLINES_RE = re.compile(r"\n(?!\u)")  # "\n" followed by any character that is not a unicode character
# ERRONEOUS_NEWLINES_RE = re.compile(ur"\n(?!\\u[0-9a-f])")  # "\n" followed by any character that is not a unicode character
def main(
    model_id,
    token_chunk_len,
    min_word_per_chunk):

    # Setting up logging
    logging = setup_logging_preprocess( 'rsearchpaper', model_id )

    # Locate data to be tokenized
    data_dir = os.path.join( './datasets/finetune/text_format/' )

    # Locate tokenizer to use
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Convert data to huggingface dataset    
    dataset = Dataset.from_generator( dataset_generator, gen_kwargs={'data_dir':data_dir} )

    dataset_dict = dataset.train_test_split( test_size=0.2, shuffle=True)

    # Compute average number of tokens in 200 words from a sample of your data
    sample_text = ' '.join(dataset['text'][:200])  # Create a sample text from your data
    num_tokens_in_sample = len(tokenizer.encode(sample_text))
    num_words_in_sample = len(sample_text.split())
    avg_tokens_per_word = num_tokens_in_sample / num_words_in_sample

    # Compute text_chunk_len based on average tokens per word
    # NOTE: This will underpredict due to the math formulas in uncleaned texts
    text_chunk_len = int(token_chunk_len / avg_tokens_per_word) 
    
    dataset_dict = dataset_dict.map( lambda batch: break_up_text(batch, text_chunk_len, min_word_per_chunk), batched=False )
    dataset_dict = dataset_dict.map( lambda batch: {'text': batch['text'][0] }, batched=True  )

    # Loop over dataset applying tokenizer to each row
    dataset_dict = dataset_dict.map( lambda batch: map_tokenize(batch, tokenizer, max_len=token_chunk_len), batched=True  )

    # Add labels to dataset
    dataset_dict = dataset_dict.map( lambda batch: create_labels_with_mask(batch, tokenizer), batched=True  )

    # Save Dataset in torch format
    dataset_train = dataset_dict['train']
    dataset_val = dataset_dict['test']

    dataset_train.set_format(type='torch', columns=["text", "input_ids", "attention_mask"] )
    dataset_val.set_format(type='torch', columns=["text", "input_ids", "attention_mask"] )
    
    # Saving to disk
    dir_ = f'./data/finetune/rp_{model_id.replace("/","_")}'
    os.makedirs(dir_)

    dataset_train.save_to_disk( dir_+'_train.arrow') 
    dataset_val.save_to_disk( dir_+'_val.arrow') 

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
    
    outp = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_len,)

    return outp

def create_labels_with_mask(batch, tokenizer):

    batch['input_ids'][batch['input_ids'] == tokenizer.pad_token_id] = -100

    batch['labels'] = batch['input_ids'].copy()
    batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100
    batch['labels'] = batch['labels'][1:] + [-100]  # shift labels to the left, append -100 to the end

    return batch

def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--model_id', type=str, 
                        default='TheBloke/Wizard-Vicuna-13B-Uncensored-HF'
                          )
    
    # mosaicml/mpt-7b-chat', 'JosephusCheung/Guanaco','TheBloke/stable-vicuna-7B-HF','TheBloke/stable-vicuna-13B-HF'
    parser.add_argument('--token_chunk_len',type=int, default=200 )
    parser.add_argument('--min_word_per_chunk',type=int, default=20)

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    main(**vars(args))

    



