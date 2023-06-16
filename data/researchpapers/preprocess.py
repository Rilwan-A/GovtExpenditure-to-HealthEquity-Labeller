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
from langdetect import detect, LangDetectException
from datasets import Features, Value


from prompt_engineering.my_logger import setup_logging_preprocess

NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

# ERRONEOUS_NEWLINES_RE = re.compile(r"\n(?!\u)")  # "\n" followed by any character that is not a unicode character
# ERRONEOUS_NEWLINES_RE = re.compile(ur"\n(?!\\u[0-9a-f])")  # "\n" followed by any character that is not a unicode character
def main(
    model_id,
    max_tokens_per_chunk,
    min_tokens_per_chunk,
    prop_chunk_overlap,
    languages_to_include:list[str]
    ):

    # Setting up logging
    logging = setup_logging_preprocess( 'rsearchpaper', model_id )

    # Locate data to be tokenized
    data_dir = os.path.join( './data/researchpapers/text_format/' )

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

    # Compute max_tokens_per_chunk based on average tokens per word
    # NOTE: This will underpredict due to the math formulas in the texts
    max_words_per_chunk = int(max_tokens_per_chunk / avg_tokens_per_word)
    min_words_per_chunk = int(min_tokens_per_chunk / avg_tokens_per_word) 
    
    dataset_dict = dataset_dict.map( 
        lambda batch: break_up_text(batch, max_words_per_chunk, min_words_per_chunk, prop_chunk_overlap), 
        batched=True,
        batch_size=300,
        remove_columns=dataset_dict.column_names['train']
    )
    
    # Filter based on language
    dataset_dict = dataset_dict.map( lambda batch: filter_on_language(batch, languages_to_include), batched=True, batch_size=500, 
                                    remove_columns=dataset_dict.column_names['train'] )

    # Loop over dataset applying tokenizer to each row
    dataset_dict = dataset_dict.map( lambda batch: map_tokenize(batch, tokenizer, max_len=max_tokens_per_chunk), batched=True, batch_size=500
                                     )

    # Add labels to dataset
    dataset_dict = dataset_dict.map( lambda batch: create_labels_with_mask(batch, tokenizer), batched=False  )

    # Save Dataset in torch format
    dataset_train = dataset_dict['train']
    dataset_test = dataset_dict['test']

    dataset_train.set_format(type='torch', columns=["text", "input_ids", "attention_mask"] )
    dataset_test.set_format(type='torch', columns=["text", "input_ids", "attention_mask"] )
    
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

        # with gz.open(fp, "rb") as f:
            # text = f.read()
            # text = text.decode('utf-8')
        with open(fp, "r") as f:
            text = f.read()
            yield {'text':text}

def break_up_text(dict_text:Dict[str,str], max_words_per_chunk:int=200, min_words_per_chunk:int=10, prop_chunk_overlap:float=0.25):
    
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

def map_tokenize(batch, tokenizer, max_len:int):
    # Tokenize each row of the dataset
    # batch['text'] is a list of strings
    
    outp = tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_len)

    return outp

def create_labels_with_mask(batch, tokenizer):

    # Create labels for each token

    batch['labels'] = [-100]*len(batch['input_ids']) 

    if tokenizer.eos_token_id in batch['input_ids']:
        eos_token_idx = batch['input_ids'].index(tokenizer.eos_token_id) 
        batch['labels'][:eos_token_idx+1] = batch['input_ids'][:eos_token_idx+1]  # set labels to input_ids
        batch['labels'] = batch['labels'][1:] + [-100]  # shift labels to the left, append -100 to the end
    else:
        batch['labels'] = batch['input_ids']
        batch['labels'] = batch['labels'][1:] + [-100]  # shift labels to the left, append -100 to the end

    return batch

def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--model_id', type=str, 
                        default='TheBloke/Wizard-Vicuna-13B-Uncensored-HF')
    
    # mosaicml/mpt-7b-chat', 'JosephusCheung/Guanaco','TheBloke/stable-vicuna-7B-HF','TheBloke/stable-vicuna-13B-HF'
    parser.add_argument('--max_tokens_per_chunk',type=int, default=500 )
    parser.add_argument('--min_tokens_per_chunk',type=int, default=100)
    
    parser.add_argument('--languages_to_include', nargs='+', default=['en'], choices=['en','es'], help='List of languages to filter for')

    parser.add_argument('--prop_chunk_overlap',type=float, default=0.35, help='Number of tokens to overlap between chunks')

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    main(**vars(args))

    



