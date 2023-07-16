# helper.py
import os
import sys

from prompt_engineering.langchain.utils import load_llm
import torch
    
# Define a function to run the first command
def run_command(queue, prediction_generator, li_li_filledtemplate, llm_name, device, method):

    # Redirect stdout and stderr to devnull
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
        
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.cuda.set_device(device)
    
    llm = load_llm(llm_name, False, 'local')
    prediction_generator.llm = llm

    # Run the first command
    li_preds= prediction_generator.aggregate_predictions( prediction_generator.predict(li_li_filledtemplate) )
    
    # Put the result in the queue
    queue.put({'name': method,
               'li_preds':li_preds})

# # Define a function to run the second command
# def run_second_command(queue):
#     # Set the CUDA_VISIBLE_DEVICES environment variable to GPU 2
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
#     # Run the second command
#     li_preds_reverse = prediction_generator.aggregate_predictions( prediction_generator.predict(li_li_filledtemplate_reverse, reverse_categories=True) )
    
#     # Put the result in the queue
#     queue.put(li_preds_reverse)
