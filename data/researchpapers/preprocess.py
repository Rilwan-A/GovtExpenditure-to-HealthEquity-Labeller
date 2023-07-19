import os
import sys
sys.path.append(os.getcwd())

from argparse import ArgumentParser
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from torch.cuda import empty_cache
import gc
import glob
import gzip as gz

import regex as re
from more_itertools import windowed
from datasets import Dataset
from typing import Dict
from langdetect import detect, LangDetectException
from datasets import Features, Value

from prompt_engineering.my_logger import setup_logging_preprocess
from prompt_engineering.langchain.utils import load_llm
from prompt_engineering.utils_prompteng import map_llmname_input_format

import multiprocessing as mp
import math
import re

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

# ERRONEOUS_NEWLINES_RE = re.compile(r"\n(?!\u)")  # "\n" followed by any character that is not a unicode character
# ERRONEOUS_NEWLINES_RE = re.compile(ur"\n(?!\\u[0-9a-f])")  # "\n" followed by any character that is not a unicode character
def main(
    model_id,
    max_tokens_per_chunk,
    min_tokens_per_chunk,
    prop_chunk_overlap,
    languages_to_include:list[str],
    split_combined_words:bool,
    debugging:bool,
    ):

    # Setting up logging
    logging = setup_logging_preprocess( 'rsearchpaper', model_id )

    # Locate data to be tokenized
    data_dir = os.path.join( './data/researchpapers/text_format/' )

    # Locate tokenizer to use
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Convert data to huggingface dataset    
    dataset = Dataset.from_generator( dataset_generator, gen_kwargs={'data_dir':data_dir} )

    # select 10 samples for debugging
    if debugging:
        dataset = dataset.select(range(2))

    dataset_dict = dataset.train_test_split( test_size=0.2, shuffle=True if (not debugging) else False)

    # Compute average number of tokens in 200 words from a sample of your data
    if not debugging:
        sample_text = ' '.join(dataset['text'][:200])  # Create a sample text from your data
        num_tokens_in_sample = len(tokenizer.encode(sample_text))
        num_words_in_sample = len(sample_text.split())
        avg_tokens_per_word = num_tokens_in_sample / num_words_in_sample
    else:
        avg_tokens_per_word = 2.0

    # Compute max_tokens_per_chunk based on average tokens per word
    # NOTE: This will underpredict due to the math formulas in the texts
    max_words_per_chunk = int(max_tokens_per_chunk / avg_tokens_per_word)
    min_words_per_chunk = int(min_tokens_per_chunk / avg_tokens_per_word) 
    
    dataset_dict = dataset_dict.map( 
        lambda batch: split_text_parallel(batch, max_words_per_chunk, min_words_per_chunk, prop_chunk_overlap), 
        batched=True,
        batch_size=700,
        remove_columns=dataset_dict.column_names['train']
    )
    
    # Filter based on language
    dataset_dict = dataset_dict.map( lambda batch: filter_on_language_parallel(batch, languages_to_include), batched=True, batch_size=700, 
                                    remove_columns=dataset_dict.column_names['train'] )
    
    # Fixing lack of spaces between parts of text
    if split_combined_words:
        llm = load_llm( 'TheBloke/wizard-vicuna-13B-HF', False, 'local', 0 )
        llm.pipeline._forward_params['max_new_tokens'] = max_tokens_per_chunk
        
        dataset_dict = dataset_dict.map( lambda batch: fix_text(batch, llm), batched=True, batch_size=500, remove_columns=dataset_dict.column_names['train'], num_proc=1)
        # release the memory used by the llm
        del llm
        gc.collect()
        empty_cache()


    # Loop over dataset applying tokenizer to each row
    dataset_dict = dataset_dict.map( lambda batch: map_tokenize(batch, tokenizer, max_len=max_tokens_per_chunk), batched=True, batch_size=500
                                     )

    # Add labels to dataset
    dataset_dict = dataset_dict.map( lambda batch: create_labels_with_mask(batch, tokenizer), batched=False )

    # Save Dataset in torch format
    dataset_train = dataset_dict['train']
    dataset_test = dataset_dict['test']

    dataset_train.set_format(type='torch', columns=["input_ids", "attention_mask", "labels"] )
    dataset_test.set_format(type='torch', columns=["input_ids", "attention_mask", "labels"] )
    
    # Saving to disk
    dir_ = f'./data/finetune'
    os.makedirs(dir_, exist_ok=True)

    dataset_train.save_to_disk( os.path.join(dir_,f'rp_{model_id.replace("/","_")}_train.arrow')) 
    dataset_test.save_to_disk( os.path.join(dir_,f'rp_{model_id.replace("/","_")}_test.arrow'))

    logging.info('Finished Preprocessing Data')
    
    return None

def dataset_generator(data_dir:str):
    
    gen_fp = glob.glob( os.path.join(data_dir,'**','[0-9]'*3+'.txt' ) )

    for fp in gen_fp:

        with open(fp, "r") as f:
            text = f.read()
            yield {'text':text}


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def split_text(dict_text:Dict[str,str], max_words_per_chunk:int=200, min_words_per_chunk:int=10, prop_chunk_overlap:float=0.25):
    
    # Split text into chunks of text with length M
    # One research paper, can now become i>1 input datums
    batched_text = dict_text['text']
    
    li_text_chunked = []

    for text in batched_text:
        # First split by paragraph / section
        text_split:list[str] = split_paragraphs(text, min_words_per_chunk)

        # For each txt in text_split, Splitting based on max_len, with overlap
        text_split_split = [ list(map( lambda seq:  ' '.join(seq).strip(' '),
                            windowed( txt.split(' ') , max_words_per_chunk, step=int(max_words_per_chunk*(1-prop_chunk_overlap)),fillvalue='' ) 
                            ))
                            for txt in text_split ] 
        
        # flatten and add to li_text_chunked
        li_text_chunked.extend(sum(text_split_split, []))

    
    return {'text':li_text_chunked}

def split_text_parallel(dict_text: Dict[str, str], max_words_per_chunk: int = 200, min_words_per_chunk: int = 10, prop_chunk_overlap: float = 0.25):
    
    
    procs=min(mp.cpu_count(),6)
    inp_batch_text = list(chunks(dict_text['text'], math.ceil(len(dict_text['text'])/procs)))

    with mp.Pool( procs ) as p:
        li_text_chunked = p.starmap(_split_text_parallel, [(texts, max_words_per_chunk, min_words_per_chunk, prop_chunk_overlap) for texts in inp_batch_text])

    # flatten li_text_chunked
    li_text_chunked = [text for batch in li_text_chunked for text in batch]


    return {'text':li_text_chunked}

def _split_text_parallel(texts: str, max_words_per_chunk: int, min_words_per_chunk: int, prop_chunk_overlap: float):
    
    outp_batch_text = []

    for text in texts:
    
        # First split by paragraph / section
        text_split: list[str] = split_paragraphs(text, min_words_per_chunk)

        # For each txt in text_split, Splitting based on max_len, with overlap
        text_split_split = [ list(map( lambda seq:  ' '.join(seq).strip(' '),
                            windowed( txt.split(' ') , max_words_per_chunk, step=int(max_words_per_chunk*(1-prop_chunk_overlap)),fillvalue='' ) 
                            ))
                            for txt in text_split ] 
        
        # flatten and add to li_text_chunked
        text_split = sum(text_split_split, [])

        outp_batch_text.extend(text_split)
    
    return outp_batch_text

def split_paragraphs(input_text:str="", min_words_per_chunk:int=10):
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
    split_text = [txt for txt in split_text if len(txt.split(' '))>= min_words_per_chunk ]

    # split_text = [p + "\n" for p in split_text if p.strip()]
    # p + "\n" ensures that all lines in the paragraph end with a newline
    # p.strip() == True if paragraph has other characters than whitespace

    return split_text


def filter_on_language(batch, languages_to_include:list[str]=['en']):
    """
    Filter out text that is not in the languages_to_include list
    Also removes text where the language can not be discerned; usually implies gibberish
    """

    inp_batch_text = batch['text']
    outp_batch_text = []

    for text in inp_batch_text:
        try:
            lang = detect(text)
            if lang in languages_to_include:
                outp_batch_text.append(text)
        except LangDetectException as e:
            pass
    
    return {'text':outp_batch_text}

def filter_on_language_parallel(batch: Dict[str, list[str]], languages_to_include: list[str] = ['en']):
    procs=min(mp.cpu_count(),6)

    inp_batch_text = list(chunks(batch['text'], math.ceil(len(batch['text'])/procs)))
        
    with mp.Pool( procs ) as p:
        outp_batches_text = p.starmap(_filter_on_language_parallel, [(texts, languages_to_include) for texts in inp_batch_text])

    # Flatten list of batches
    outp_batch_text = [text for batch in outp_batches_text for text in batch]

    return {'text': outp_batch_text}

def _filter_on_language_parallel(texts: list[str], languages_to_include: list[str]):
    outp_batch_text = []

    for text in texts:
        try:
            lang = detect(text)
            if lang in languages_to_include:
                outp_batch_text.append(text)
        except LangDetectException:
            pass

    return outp_batch_text


def fix_text(batch, llm):
    """ 
        1) First remove unicode control characters
        2) Due to pdf parsing package, sometimes words are joined to gether in parsed text"""

    # Part 1)
    texts = []
    for text in batch['text']:
        text = remove_unicode_directionality(text)
        texts.append(text)

    # Part 2)
    li_split_text = []
    user_message = 'Please fix grammatical and lexical mistakes in the following text. Start the corrected text with the phrase, "Corrected Text:". \nText:'
    li_prompts_fmtd = [
        map_llmname_input_format(llm.model_id, 
                                                user_message = user_message + ' ' + text,
                                                system_message = None) for text in texts
    ]

    outputs = llm.generate(
                prompts=li_prompts_fmtd,
    )

    # remove any double spaces on ends
    li_preds : list[str] = [ chatgen.text.strip(' ') for chatgen in sum(outputs.generations,[]) ]

    for text in li_preds:
        # Check if text starts with 'Corrected Text:'
        if not text.startswith('Corrected Text:'):
            continue

        # strip all text before and including 'Corrected Text:',
        text = text.split('Corrected Text:')[-1]
        # remove any double spaces / new lines are start or end of text
        text = text.strip(' \n')

        li_split_text.append(text)

    return {'text':li_split_text}

def remove_unicode_directionality(s):
    s = re.sub(r'[\u202B\u202A\u202C]\n?', '', s)
    return s

def map_tokenize(batch, tokenizer, max_len:int):
    # Tokenize each row of the dataset
    # batch['text'] is a list of strings
    tokenizer.padding_side = 'right'
    tokenizer.truncation_side = 'right'
    outp = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_len )

    return outp

def create_labels_with_mask(batch, tokenizer):

    # Create labels for each token

    labels = [-100]*len(batch['input_ids']) 

    # Find the index where the first bos appears in the input_ids
    idx_first_bos = batch['input_ids'].index(tokenizer.bos_token_id)

    if tokenizer.eos_token_id in batch['input_ids'][idx_first_bos:]:
        eos_token_idx = batch['input_ids'][idx_first_bos:].index(tokenizer.eos_token_id) + idx_first_bos
        labels[:eos_token_idx+1] = batch['input_ids'][:eos_token_idx+1]  # set labels to input_ids
        labels = labels[1:] + [-100]  # shift labels to the left, append -100 to the end
    else:
        labels = batch['input_ids']
        labels = labels[1:] + [-100]  # shift labels to the left, append -100 to the end

    return {'labels':labels}

def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--model_id', type=str, 
                        default='TheBloke/Wizard-Vicuna-13B-Uncensored-HF')
    
    # mosaicml/mpt-7b-chat', 'JosephusCheung/Guanaco','TheBloke/stable-vicuna-7B-HF','TheBloke/stable-vicuna-13B-HF'
    parser.add_argument('--max_tokens_per_chunk',type=int, default=500 )
    parser.add_argument('--min_tokens_per_chunk',type=int, default=100)
    
    parser.add_argument('--languages_to_include', nargs='+', default=['en'], choices=['en','es'], help='List of languages to filter for')

    parser.add_argument('--prop_chunk_overlap', type=float, default=0.35, help='Number of tokens to overlap between chunks')

    parser.add_argument('--split_combined_words', action='store_true', help='Whether to split combined words', default=False)

    parser.add_argument('--debugging', action='store_true', help='Whether to run in debugging mode', default=False)

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    main(**vars(args))

    



