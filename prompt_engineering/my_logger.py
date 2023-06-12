import logging
import os
from datetime import datetime

def setup_logging(filename):
    # Create logging directory if it doesn't exist
    if not os.path.exists('logging'):
        os.makedirs('logging')

    log_filename = f'logging/{filename}'

    # Configure logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    return logging

def setup_logging_predict( llm_name ):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    llm_name = ''.join(llm_name.split('/')[1:])

    log_filename = f'{llm_name}_{dt_string}.log'

    logging = setup_logging(log_filename)
    return logging

