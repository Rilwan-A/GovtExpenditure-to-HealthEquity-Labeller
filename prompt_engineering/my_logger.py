import logging
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import os, sys
from datetime import datetime

def setup_logging(filename, debugging=False):
    if not debugging:
        # Create logging directory if it doesn't exist
        if not os.path.exists('logging'):
            os.makedirs('logging')

        log_filename = f'logging/{filename}'
    else:
        log_filename = None

    # Configure logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    return logging

def setup_logging_predict( llm_name, debugging=False ):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    llm_name = ''.join(llm_name.split('/')[1:])

    log_filename = f'{llm_name}_{dt_string}.log'

    logging = setup_logging(log_filename, debugging)

    sys.excepthook = lambda exctype, value, traceback: logging.exception(value)

    return logging


def setup_logging_preprocess( dset_name, llm_name ):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")

    llm_name = ''.join(llm_name.split('/')[1:])

    log_filename = f'pprcs_{dset_name}_{llm_name}_{dt_string}.log'

    logging = setup_logging(log_filename)

    sys.excepthook = lambda exctype, value, traceback: logging.exception(value)

    return logging

