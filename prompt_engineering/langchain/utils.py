import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain import PromptTemplate, LLMChain

HUGGINGFACE_MODELS = [ 'mosaicml/mpt-7b-instruct' ]
OPENAI_MODELS = ['gpt-3.5-turbo-030', 'gpt-4']

# chat_models = ['gpt-3.5-turbo-030', 'gpt-4', 'mosaicml/mpt-7b-instruct']

class PredictionGeneratorRemoteLM():
    """
        NOTE: This prediction generator currently only designed for models tuned on instruct datasets that are remote
    """
    def __init__(self, lm,  
                 model_type:str,
                 prompt_style:str,
                  ensemble_size:int,
                  effect_order:str='arbitrary', 
                  edge_value:str="0/1",
                  parse_style:str='rule_based',
                  deepspeed_compat:bool=False ):
        
        
        self.lm = lm
        # discern if the langchain lm is a chat model or lm model

        self.prompt_style = prompt_style
        self.ensemble_size = ensemble_size
        self.parse_style = parse_style
        self.deepspeed_compat = deepspeed_compat

        self.aggregation_method = None
        if edge_value == '0/1':
            self.aggregation_method = 'majority_vote'
        elif effect_order == 'float':
            self.aggregation_method = 'average'


    def predict(self, li_li_prompts:list[list[str]])->tuple[list[list[str]], list[list[str]]]:
        "Given a list of prompt ensembels, returns a list of predictions, with one prediction per member of the ensemble"
        
        # Generate predictions
        li_li_preds = []
        
        for li_prompts in li_li_prompts:

            if isinstance(self.lm, langchain.chat_models.base.BaseChatModel):
            
            
            elif isinstance(self.lm, langchain.llms.base.LLM):
            
               outputs = self.lm.generate( prompts= li_prompts )
               li_preds = [ outputs[i]['text'] for i in range(len(outputs)) ]

            li_li_preds.append(outputs)

        if 

        # Extract just the answers from the decoded texts, removing the prompts
        li_li_predictions = [ [ pred.replace(prompt,'') for pred, prompt in zip(li_prediction, li_prompt) ] for li_prediction, li_prompt in zip(li_li_predictions, li_li_prompts) ]
        
        # Parse Yes/No/Nan from the predictions
        if self.parse_output_method == 'rule_based':
            li_li_predictions_parsed = [ self.parse_yesno_with_rules(li_predictions) for li_predictions in li_li_predictions]

        elif self.parse_output_method == 'language_model_perplexity':
            li_li_predictions_parsed = [ self.parse_yesno_with_lm_perplexity(li_predictions) for li_predictions in li_li_predictions]
        
        elif self.parse_output_method == 'language_model_generation':
            li_li_predictions_parsed = [ self.parse_yesno_with_lm_generation(li_predictions) for li_predictions in li_li_predictions]
        
        else:
            raise ValueError(f"parse_output_method {self.parse_output_method} not recognized")

        return li_li_predictions, li_li_predictions_parsed


    def parse_yesno_with_rules(self, li_predictions:list[str])->list[str]:
        
        li_preds_parsed = ['NA']*len(li_predictions)

        # Parse yes/no from falsetrue
        for idx in range(len(li_predictions)):
            
            prediction = li_predictions[idx].lower()

            if 'yes' in prediction:
                prediction = 'Yes'
            
            elif 'no' in prediction:
                prediction = 'No'
            
            elif any( ( neg_phrase in prediction for neg_phrase in ['not true','false', 'is not', 'not correct', 'does not', 'can not', 'not'])):
                prediction = 'No'

            else:
                prediction = 'NA'
        
            li_preds_parsed[idx] = prediction
                   
        return li_predictions
               
    def parse_yesno_with_lm_generation(self, li_predictions:list[str])->list[str]:
        
        # Template to prompt language lm to simplify the answer to a Yes/No output
        template = copy.deepcopy( utils_prompteng.li_prompts_parse_yesno_from_answer[0] )

        # Create filled versions of the template with each of the predictions
        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions]

        # Create batch encoding
        batch_encoding = self.tokenizer(li_filledtemplate, return_tensors='pt', padding=True, truncation_strategy='do_not_truncate')

        # Move to device
        batch_encoding = batch_encoding.to(self.lm.device)


        # setup generation config for parsing yesno
        eos_token_ids = [self.tokenizer.eos_token_id] + [ self.tokenizer(text)['input_ids'][-1] for text in ['"Negation".', '"Affirmation".', 'Negation.', 'Affirmation.','\n' ] ]
        
        gen_config = transformers.GenerationConfig(max_new_tokens = 20, min_new_tokens = 2, early_stopping=True, 
                                            temperature=0.7, no_repeat_ngram_size=3,
                                            eos_token_id=eos_token_ids,
                                            do_sample=False,
                                            pad_token_id = self.tokenizer.pad_token_id   )
        
        # Generate prediction
        output = self.lm.generate(**batch_encoding, generation_config=gen_config )

        # Decode output
        predictions_w_prompt = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        # Extract just the answers from the decoded texts, removing the prompts
        li_predictions = [ pred.replace(template,'') for pred,template in zip(predictions_w_prompt, li_filledtemplate) ]

        # Parse Yes/No/Na from the prediction
        li_predictions_parsed = [ 'Yes' if 'affirm' in pred.lower() else 'No' if 'negat' in pred.lower() else 'NA' for pred in li_predictions]

        return li_predictions_parsed
    
    def parse_yesno_with_lm_perplexity(self, li_predictions:list[str])->list[str]:
        # Get average perplexity of text when sentence is labelled Negation vs when it is labelled Affirmation.
        # NOTE: the perpleixty is calculated as an average on the whole text, not just the answer. Therefore, we rely on the
        #       the fact that 'Negation". and "Affirmation". both contain the same number of tokens

        # Template to prompt language lm to simplify the answer to a Yes/No output
        template = copy.deepcopy( utils_prompteng.li_prompts_parse_yesno_from_answer[0] )

        li_filledtemplate = [ template.format(statement=pred) for pred in li_predictions]

        # For each fill template, create 3 filled versions with each of the possible answers
        # NOTE: The answers must not include any extra tokens such as punctuation since this will affect the perplexity
        answers = ['Negation', 'Affirmation']
        li_li_filledtemplates_with_answers = [ [ filledtemplate + ' ' + ans for ans in answers ] for filledtemplate in li_filledtemplate ]
        li_filledtemplates_with_answers = sum(li_li_filledtemplates_with_answers,[])

        # Get the perplexity of each of the filled templates
        li_perplexity = utils_prompteng.perplexity(li_filledtemplates_with_answers, self.lm, self.tokenizer, batch_size=6, deepspeed_compat = self.deepspeed_compat ) 

        # For each set of filltemplates get the index of the answer with the lowest perplexity
        li_idx = [ np.argmin(li_perplexity[idx:idx+len(answers)]) for idx in range(0,len(li_perplexity),len(answers)) ]

        li_predictions = [ 'No' if idx==0 else 'Yes' for idx in li_idx ]
        
        return li_predictions
        
    def aggregate_predictions(self, li_li_predictions:list[list[str]])->list[str]:
        "Given a list of predictions, returns a single prediction"
        
        if self.ensemble_size == 1:
            li_predictions = [ li_prediction[0] for li_prediction in li_li_predictions ]
        
        elif self.aggregation_method == 'majority_vote':
            li_predictions = [ Counter(li_prediction).most_common(1)[0][0] for li_prediction in li_li_predictions ]
        
        else:
            raise NotImplementedError(f"Aggregation method {self.aggregation_method} not implemented")
        
        return li_predictions

def load_annotated_examples(k_shot_example_dset_name:str, 
                            random_state_seed:int=10, 
                            relationship_type:str='budget_item_to_indicator') -> list[dict]:
    
    if k_shot_example_dset_name == 'spot' and relationship_type == 'budget_item_to_indicator':
        # Load spot dataset as pandas dataframe
        dset = pd.read_csv('./data/spot/spot_indicator_mapping_table.csv')
        
        # Remove all rows where 'type' is not 'Outcome'
        dset = dset[dset['type'] == 'Outcome']

        # Creating target field
        dset['label'] = 'Yes'

        # Rename columns to match the format of the other datasets
        dset = dset.rename( columns={'category': 'budget_item', 'name':'indicator' } )

        # Create negative examples
        random_state = np.random.RandomState(random_state_seed)

        # Too vague a budget item and there are only 4 examples of it, we remove it
        dset = dset[ dset['budget_item'] != 'Central' ]

        # create negative examples
        dset = utils_prompteng.create_negative_examples(dset, random_state=random_state )

        # Removing rows that can not be stratified due to less than 2 unique examples of budget_item and label combination
        dset = dset.groupby(['budget_item','label']).filter(lambda x: len(x) > 1)

        li_records = dset.to_dict('records')
    
    elif k_shot_example_dset_name == 'spot' and relationship_type == 'indicator_to_indicator':
        logging.log.warning('Currently, there does not exist any annotated examples for indicator to indicator relationships. Therefore we can not use K-Shot templates for indicator to indicator edge determination. This will be added in the future')        
        li_records = None

    elif k_shot_example_dset_name == 'england':
        logging.log.warning('Currently, the relationshps for England dataset have not yet been distilled. This will be added in the future')        
        li_records = None
    
    else:
        raise ValueError('Invalid dset_name: ' + k_shot_example_dset_name)

    return li_records