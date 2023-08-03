import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import bitsandbytes as bnb
from sklearn.metrics import accuracy_score
from argparse import Namespace

from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import glob
from torch.utils.data import DataLoader, Dataset as TorchDataset
import yaml
from datasets import Dataset  # type: ignore
import torch
from argparse import ArgumentParser
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch as pl
import transformers
import logging
logging.getLogger("transformers").setLevel(logging.CRITICAL)

from sklearn.metrics import precision_recall_fscore_support

from prompt_engineering.langchain.utils import PredictionGenerator
from prompt_engineering.utils_prompteng import PromptBuilder

from transformers import BitsAndBytesConfig

from peft import get_peft_config, prepare_model_for_int8_training, get_peft_model, LoraConfig, TaskType

from prompt_engineering.langchain.utils import HUGGINGFACE_MODELS
from datasets import interleave_datasets, load_dataset
from prompt_engineering.langchain.predict import prepare_data_b2i, predict_batches

from transformers import get_constant_schedule_with_warmup
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# TODO: Add the other models to this dict mebe
map_modelid_targetmodule = {
    'TheBloke/Wizard-Vicuna-7B-Uncensored-HF': ['k_proj', 'v_proj'],
    'TheBloke/Wizard-Vicuna-13B-Uncensored-HF': ['k_proj', 'v_proj']
}

class PromptEngineeringLM(pl.LightningModule):
    """LightningModule for prompt engineering LM training and evaluation."""
    def __init__(self,
                 model,
                 tokenizer,
                 val_tasks:list[str],
                 **kwargs
                 ):
        super().__init__()
        # self.model_id = model_id

        # NOTE: Model should be able to be loaded locally, if it is already trained
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.val_tasks = val_tasks
        self.optimizer = kwargs.get('optimizer', 'auto')
        self.lr = kwargs.get('lr', 1e-6)
    
        if 'spot_alignment' in self.val_tasks:
            self.val_step_outputs_spotalign = []
            self.prompt_builder_b2i = PromptBuilder( self.model,
                                                     self.model.name_or_path,  
                                                     'yes_no',
                                                     k_shot=0,
                                                     ensemble_size=1, 
                                                     examples_dset=None, 
                                                     effect_type='directly',
                                                     relationship='budgetitem_to_indicator',
                                                     seed=kwargs.get('seed', 10),
                                                     tokenizer=self.tokenizer)
    
            self.prediction_generator_b2i = PredictionGenerator(self.model,
                                                            self.model.name_or_path,
                                                            prompt_style='yes_no',
                                                            ensemble_size=1,
                                                            edge_value="binary_weight",
                                                            parse_style='rules',
                                                            relationship='budgetitem_to_indicator',
                                                            local_or_remote='local',
                                                            effect_type='directly',
                                                            tokenizer=self.tokenizer,
                                                            )

    def forward(self, **kwargs):
        return self.model( **kwargs)

    def training_step(self, batch, batch_idx):
        batch['labels'] = batch['input_ids'].clone()
        outputs = self(**batch, output_hidden_states=False, output_attentions=False)
        loss = outputs.loss

        if torch.isnan(loss):
            return None
        
        self.log('train_loss', loss, on_step=True,
                  on_epoch=False, prog_bar=True, logger=True,
                  sync_dist=False, rank_zero_only=True)
        
        # if a nan in loss then return None so pytorch lightning skips the step
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        if self.val_tasks[dataloader_idx] == 'next_token':
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            # labels = batch['labels']
            labels = batch['input_ids'].clone()


            with torch.no_grad():
                outputs = self(input_ids=input_ids, attention_mask = attention_mask, labels=labels,
                            output_hidden_states=False, output_attentions=False)
            loss = outputs.loss
            if torch.isnan(loss):
                return None
            self.log('val_nt/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

            outp = None

        elif self.val_tasks[dataloader_idx]  == 'spot_alignment':
            # batch will be a list of dicts with keys: budget_item,id,indicator,type,label
            with torch.no_grad():
                li_prompt_ensemble, li_pred_ensemble, li_pred_agg, li_discourse_ensemble = predict_batches(
                    prompt_builder=self.prompt_builder_b2i,
                    prediction_generator=self.prediction_generator_b2i,
                    li_record = batch,
                    unbias_categorisations=False,
                    batch_size=len(batch) )

            # Convert the li_pred_agg, which is a list of agg
            li_pred_agg = [ next( (k for k,v in d.items() if v==1) ) for d in li_pred_agg]

            outp = {'pred_agg': li_pred_agg,
                    'related': [d['related'] for d in batch],
                    'indicator': [d['indicator'] for d in batch],
                    'budget_item': [d['budget_item'] for d in batch],
                    'discourse': li_discourse_ensemble
                    }

            self.val_step_outputs_spotalign.append(outp)

            outp = None
        
        else:
            raise ValueError(f'val_task {self.val_tasks} is not recognized')

        return outp

    def on_validation_epoch_end(self):

        if 'spot_alignment' in self.val_tasks:
            preds_agg = sum([d['pred_agg'] for d in self.val_step_outputs_spotalign], [])
            labels = sum([d['related'] for d in self.val_step_outputs_spotalign], [])

            # compute metrics between binary labels and predictions
            (prec_yes, prec_no), (recall_yes, recall_no), (f1_yes, f1_no), _ = precision_recall_fscore_support(labels, preds_agg, labels=['Yes', 'No'], average=None, zero_division=0)
            
            acc = accuracy_score(labels, preds_agg)


            # self.log('val_loss', -acc, on_step=False, on_epoch=True,
            #         prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)
            
            # Logging General Metrics
            self.log('val_spot/acc', acc, on_step=False, on_epoch=True,
                    prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)
            
            # Logging positive class metrics
            self.log('val_spot/f1/pos', f1_yes, on_step=False, on_epoch=True,
                    prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
            self.log('val_spot/prec/pos', prec_yes, on_step=False, on_epoch=True,
                    prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
            self.log('val_spot/rec/pos', recall_yes, on_step=False, on_epoch=True,
                    prog_bar=False, logger=True,  rank_zero_only=False, sync_dist=True)

            # Logging negative class metrics
            self.log('val_spot/f1/neg', f1_no, on_step=False, on_epoch=True,
                    prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
            self.log('val_spot/prec/neg', prec_no, on_step=False, on_epoch=True,
                    prog_bar=False, logger=True,  rank_zero_only=False, sync_dist=True)
            self.log('val_spot/rec/neg', recall_no, on_step=False, on_epoch=True,
                    prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
            
            # Saving the predictions to file
            df = pd.DataFrame({'pred_agg': preds_agg, 
                               'label': labels, 
                               'budget_item': [ bi for d in self.val_step_outputs_spotalign for bi in d['budget_item']],
                                'indicator': [ i for d in self.val_step_outputs_spotalign for i in d['indicator'] ]
                                 })
            # make a csv path that is a combination of current epoch and absolute step
            save_path = os.path.join( os.path.dirname(self.trainer.checkpoint_callback.dirpath), 'val_outputs' ,f'val_preds_epoch_{self.current_epoch}_step_{self.global_step}.csv')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv( save_path, index=False)

            self.val_step_outputs_spotalign.clear()
            
        else:
            pass
        
        self.val_dataloader = self.trainer.datamodule.val_dataloader()
        return None
    
    def configure_optimizers(self):
        # optimal params for Adafactor https://github.com/huggingface/transformers/pull/10526#issuecomment-804652154

        if self.optimizer == 'Adam8bit':
            optimizer = bnb.optim.Adam8bit(
                    self.trainer.model.model.parameters(), 
                    # lr=self.lr, 
                    betas=(0.9, 0.95),
                    eps=1e-5,
                    )
            # return {'optimizer': optimizer, 'monitor': 'val_loss'}
            return {'optimizer': optimizer, 'monitor': 'val_loss/acc'}
        
        
        elif self.optimizer == 'Adam8bitPaged':
            optimizer = bnb.optim.PagedAdamW8bit(self.parameters(), 
                                                 betas=(0.9, 0.95),
                                                    eps=1e-5,
                                                 lr=self.lr)
            #Add a CosineLR scheduler
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
            #                                                        T_max= se, 
            #                                                        eta_min=self.lr/1000 ) 

            # add warmup
            scheduler = get_constant_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=25)

            # return {'optimizer': optimizer, 'monitor': 'val_loss', 'lr_scheduler': scheduler}
            return {'optimizer': optimizer, 'monitor': 'val_loss/acc', 'lr_scheduler': scheduler}

        else:
            raise ValueError(f'optimizer {self.optimizer} not recognized')

    @staticmethod
    def train_model(config_trainer, config_data, config_model):

        # Creating Model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        
        model = transformers.AutoModelForCausalLM.from_pretrained(config_trainer.model_id,
                                                                trust_remote_code=True,
                                                                quantization_config=bnb_config,
                                                                # device_map={'':0},  
                                                                  device_map = 'auto'
                                                                )
        
        # Implementing Lora version
        peft_config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=map_modelid_targetmodule[config_trainer.model_id], 
            lora_dropout=0, 
            bias="none", 
            inference_mode=False,
            task_type=TaskType.CAUSAL_LM
        )
        # prepare int-8 model for training
        # model = prepare_model_for_int8_training(model)

        model = get_peft_model(model, peft_config)
        print_trainable_parameters(model)

        # Creating Tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config_trainer.model_id, use_fast=True, )

        # Create data module
        data_module = DataModule(**vars(config_data))

        # Create training module
        lightning_module = PromptEngineeringLM(
            model, tokenizer, val_tasks=config_data.val_tasks, **vars(config_trainer))

        # Trainer Callbacks
        callbacks = []
        
        callbacks.append(
            ModelCheckpoint(
                monitor="val_nt/loss",
                filename='epoch={epoch}-step-{step}-val_nt_loss={val_nt/loss:.3f}',
                save_last=False,
                auto_insert_metric_name=False,
                save_weights_only=True,
                save_top_k=1))
        
        callbacks.append(
            ModelCheckpoint(
                monitor="val_spot/acc",
                filename='epoch={epoch}-step={step}-val_spot_acc={val_spot/acc:.3f}',
                save_last=False,
                auto_insert_metric_name=False,
                save_weights_only=True,
                save_top_k=2))
        
        callbacks.append(
            PairedEarlyStopping(
                                patience_1=config_trainer.patience_1,
                                patience_2=config_trainer.patience_2, 
                                    monitor_1='val_nt/loss',
                                    monitor_2='val_spot/acc',
                                    mode_1='min',
                                    mode_2='max' )
                                    )
        # callbacks.append(
        #     EarlyStopping(monitor='val')
        #     )

        # setting up strategy
        strategy = config_trainer.strategy


        # Create trainer
        trainer = pl.Trainer(
            strategy=strategy,

            accelerator=config_trainer.accelerator,
            devices=config_trainer.devices,

            callbacks=callbacks,
            logger=pl_loggers.TensorBoardLogger(
                config_trainer.dir_ckpt,
                name=config_trainer.exp_name,               
                ),
            precision= 32,
            accumulate_grad_batches=config_trainer.accumulate_grad_batches,

            max_epochs=5 if config_trainer.debugging else config_trainer.max_epochs,
            num_sanity_val_steps=0,
            
            limit_train_batches=1 if config_trainer.debugging else None,
            # limit_val_batches=2 if config_trainer.debugging else None,
            log_every_n_steps=1 if config_trainer.debugging else 10,
            
            val_check_interval=config_trainer.val_check_interval
            )

        # Train model
        for p in lightning_module.model.parameters():
            p = p.contiguous()
        trainer.fit(lightning_module, data_module)

        # Saving configs relating to experiment
        os.makedirs(trainer.checkpoint_callback.dirpath, exist_ok=True)
        yaml.dump(vars(config_trainer), open(os.path.join(
            trainer.checkpoint_callback.dirpath, 'config_trainer.yaml'), 'w'))  # type: ignore
        yaml.dump(vars(config_data), open(os.path.join(
            trainer.checkpoint_callback.dirpath, 'config_data.yaml'), 'w'))  # type: ignore
        yaml.dump(vars(config_model), open(os.path.join(
            trainer.checkpoint_callback.dirpath, 'config_model.yaml'), 'w'))  # type: ignore
    
    @staticmethod
    def test_model(config_trainer, config_data, config_model=None):
        """Test model on test set"""
        # Get paths for saved model
        dir_ckpt = config_trainer.dir_ckpt if config_trainer.dir_ckpt else ''
        dir_model = os.path.join(dir_ckpt, f"{config_trainer.exp_name}")
        dir_model_version = os.path.join(
            dir_model, "lightning_logs", f"version_{config_trainer.test_version}")

        # Load configs
        config_trainer = yaml.safe_load(
            open(os.path.join(dir_model_version, 'configs', 'config_trainer.yaml'), 'r'))
        config_data = yaml.safe_load(
            open(os.path.join(dir_model_version, 'configs', 'config_data.yaml'), 'r'))
        config_model = yaml.safe_load(
            open(os.path.join(dir_model_version, 'configs', 'config_model.yaml'), 'r'))

        # allowing user to update test parameters used
        changed_args_t = {k: getattr(config_trainer, k) for k in [
            'sample_size', 'batch_size_inf_nt'] if hasattr(config_trainer, k)}
        changed_args_d = {k: getattr(config_data, k) for k in [
            'test_start', 'test_end', 'data_dir'] if hasattr(config_data, k)}

        for k, v in changed_args_t.items():
            setattr(config_trainer, k, v)
        for k, v in changed_args_d.items():
            setattr(config_data, k, v)

        # Loading parameters for saved model
        raise NotImplementedError
        checkpoint_path = next((elem for elem in glob.glob(os.path.join(dir_model_version, "checkpoints", "*"))
                                if elem[-4:] == "ckpt"))

        early_stopping_output = torch.load(checkpoint_path)[
            'callbacks']['early_stopping']

        best_checkpoint_path = early_stopping_output['best_k_models'][0]['path']

        # Load a transformer model from checkpoint path using a pytorch lightning checkpoint
        pt_checkpoint_path = os.path.join(
            best_checkpoint_path.rstrip('.ckpt')+'_pt.ckpt', '.ckpt')
        model = transformers.AutoModelForCausalLM.from_pretrained(pt_checkpoint_path,
                                                                  load_in_8bit=True,
                                                                  device_map="auto")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config_trainer.model_id, use_fast=True)

        # Making training module
        lightning_module = PromptEngineeringLM(
            model, tokenizer, **vars(config_trainer))

        # Making DataModule
        data_module = DataModule(**vars(config_data))

        trainer = pl.Trainer(
            accelerator=config_trainer.accelerator,
            devices=config_trainer.devices,
            strategy=config_trainer.strategy,
            default_root_dir=dir_model_version,
            logger=False,
            # precision=16,
        )

        # Test the Trainer
        trainer.test(
            lightning_module,
            ckpt_path=checkpoint_path,
            dataloaders=data_module.test_dataloader())

    @staticmethod
    def parse_args(parent_parser=None):

        parser = ArgumentParser(parents=[parent_parser] if parent_parser else None, add_help=True, allow_abbrev=False)

        parser.add_argument('--exp_name', type=str, default='debug')
        parser.add_argument('--model_id', type=str, default=HUGGINGFACE_MODELS[0], choices=HUGGINGFACE_MODELS )

        parser.add_argument('--test_only', action='store_true', help='Include flag to test only')
        parser.add_argument('--debugging', action='store_true')
        

        # parser.add_argument('--dir_ckpt', type=str, default='./prompt_engineering/finetune/ckpt')
        parser.add_argument('--dir_ckpt', type=str, default='/mnt/Data1/akann1w0w1ck/AlanTuring/prompt_engineering/finetune/ckpt')

     
        parser.add_argument('--optimizer', type=str,
                            choices=['Adam8bitPaged', 'Adam8bit'], 
                            default='Adam8bitPaged' )
        parser.add_argument('--lr', type=float, default=1e-4)

        
        parser.add_argument('--devices', type=int, default=1)
        parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu'])
        parser.add_argument('--strategy', type=str, default='auto',
                            choices=['auto', 'ddp', 'fsdp', 'deepspeed_stage_2', 'deepspeed_stage_2_offload',
                                     'deepspeed_stage_3', 'deepspeed_stage_3_offload'])        
        parser.add_argument('--patience_1', type=int, default=4)
        parser.add_argument('--patience_2', type=int, default=6)
        
        parser.add_argument('--accumulate_grad_batches', type=int, default=4)
        parser.add_argument('--max_epochs', type=int, default=2)
        parser.add_argument('--val_check_interval', type=lambda x: float(x) if '.' in x else int(x),  default=None )
    
        args = parser.parse_known_args()[0]

        return args

class DataModule(pl.LightningDataModule):

    def __init__(self, model_id, dir_data, batch_size, 
                    batch_size_inf_nt, batch_size_inf_sa, train_dset_names, val_tasks,
                    num_workers=4, seed=10, val_nt_batch_count=None):
        super().__init__()
        self.model_id = model_id
        self.dir_data = dir_data
        self.num_workers = num_workers

        self.batch_size = batch_size
        self.batch_size_inf_nt = batch_size_inf_nt
        self.batch_size_inf_sa = batch_size_inf_sa

        self.train_dsets_names:list[str] = train_dset_names
        self.val_tasks:list[str] = val_tasks
        self.seed = seed

        self.val_nt_batch_count = val_nt_batch_count

    def train_dataloader(self):
        # Load each of the datasets listed in self.train_dsets_names
        
        datasets = []

        if 'research_paper' in self.train_dsets_names:
            dataset = Dataset.load_from_disk(os.path.join(
                self.dir_data, 'researchpapers', 'preprocessed' ,f"rp_{self.model_id.replace('/','_')}_train.arrow"))
            datasets.append(dataset)
            # NOTE: we assume that we do not have to use a chat style 

        if 'wizardLM_instruct_70k' in self.train_dsets_names:
            dataset = Dataset.load_from_disk(os.path.join(
                self.dir_data, 'instruct', 'preprocessed' , f"wLM70k_nofilt_{self.model_id.replace('/','_')}_train.arrow"))
            datasets.append(dataset)

        # datasets interleave two datasets in a 2:1 proportion
        if len(datasets) > 1:
            dataset = interleave_datasets(datasets, 
                                        #   probabilities=[0.5, 0.5], 
                                          seed=self.seed, 
                                          stopping_strategy="first_exhausted" )
        else:
            dataset = datasets[0]

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers, 
                                pin_memory=False, drop_last=False)
        
        return dataloader

    def val_dataloader(self):
        # The validation datasets loaded is dependent on the val_tasks specified
        # If 'next_token' task is specified then we load each dataset specified in self.train_dsets_names and eval using loglikelihood on next token
        # If 'spot_alignment' task is specified then we load the spot dataset and evaluate the model's ability to produce predictions aligning with the spot dataset

        dataloaders = []
        
        if 'next_token' in self.val_tasks:

            datasets = []
            # Add each of the datasets listed in self.train_dsets_names
            if 'research_paper' in self.train_dsets_names:
                dataset = Dataset.load_from_disk(os.path.join(
                    self.dir_data, 'researchpapers', 'preprocessed', f"rp_{self.model_id.replace('/','_')}_test.arrow"))
                datasets.append(dataset)
                # NOTE: we assume that we do not have to use a chat style 

            if 'wizardLM_instruct_70k' in self.train_dsets_names:
                dataset = Dataset.load_from_disk(os.path.join(
                    self.dir_data, 'instruct', 'preprocessed' , f"wLM70k_nofilt_{self.model_id.replace('/','_')}_test.arrow"))
                datasets.append(dataset)
                
            # datasets interleave two datasets in a 1:1 proportion
            # Combine into one dataset
            if len(datasets) > 1:
                if self.val_nt_batch_count is not None:
                    li_indices  = [ torch.randperm(len(datasets[i]))[:(self.val_nt_batch_count*self.batch_size_inf_nt)//len(datasets)] for i in range(len(datasets))]
                    
                    datasets = [ dataset.select(indices) for dataset, indices in zip(datasets, li_indices) ]

                dataset = interleave_datasets(datasets, 
                                            #   probabilities=[0.5, 0.5], 
                                            #   seed=self.seed,
                                              stopping_strategy="first_exhausted")
            
            else:
                dataset = datasets[0]
                if self.val_nt_batch_count is not None:
                    indices = torch.randperm(len(dataset))[:(self.val_nt_batch_count*self.batch_size_inf_nt)]
                    dataset = dataset.select(indices)

            # Select a random subset of indicies from the dataset to evaluate on


            dataloader_nexttoken = DataLoader(dataset, batch_size=self.batch_size_inf_nt,   
                        shuffle=True, num_workers=self.num_workers//2, 
                        pin_memory=False, drop_last=False)
            
            dataloaders.append(dataloader_nexttoken)
            
        if 'spot_alignment' in self.val_tasks :

            dset_records = prepare_data_b2i(input_file=os.path.join(self.dir_data,'spot','spot_b2i_broad_test.csv'), data_load_seed=self.seed)
            
            dataloader_spotalign = DataLoader(dset_records, batch_size=self.batch_size_inf_sa,
                                    shuffle=False,
                                    num_workers=self.num_workers//2,
                                    pin_memory=False,
                                    collate_fn= lambda batch: batch,
                                    drop_last=False)
            dataloaders.append(dataloader_spotalign)
        else:
            raise ValueError(f"val_tasks {self.val_tasks} not recognized")

        return dataloaders

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
        return super().test_dataloader()

    @staticmethod
    def parse_args(parent_parser=None):
        parser = ArgumentParser(parents=[parent_parser] if parent_parser else None, add_help=True, allow_abbrev=False)

        parser.add_argument('--model_id', type=str)
        parser.add_argument('--dir_data', type=str, default='./data')

        parser.add_argument('--num_workers', type=int, default=0)

        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--batch_size_inf_nt', type=int, default=3)
        parser.add_argument('--batch_size_inf_sa', type=int, default=3)


        parser.add_argument('--train_dset_names', type=str, nargs='+', choices=[ 'research_paper', 'wizardLM_instruct_70k' ], default=['research_paper', 'wizardLM_instruct_70k'] )
                            
        parser.add_argument('--val_tasks', type=str, nargs='+', choices=['next_token', 'spot_alignment'], default=['next_token','spot_alignment'])
        parser.add_argument('--val_nt_batch_count', type=int, default=None, help='How many batches to use in validation steps for next token task')
        args = parser.parse_known_args()[0]
        return args

class PairedEarlyStopping(EarlyStopping):
    """Early Stopping that requires at least one of two metrics to improve over patience steps"""

    def __init__(self, verbose: bool = False, monitor_1: str = 'val_nt/loss', monitor_2: str = 'val_spot/acc', 
                 mode_1: str = 'min', mode_2:str = 'max',
                 patience_1: int = 3, patience_2: int = 3):
        
        super().__init__(monitor=monitor_1, patience=patience_1, verbose=verbose, mode=mode_1, check_finite=False)
        self.monitor_2 = monitor_2
        self.patience_1 = patience_1
        self.patience_2 = patience_2
        self.wait_count_1 = 0
        self.wait_count_2 = 0
        self.best_score_1 = None
        self.best_score_2 = None
        self.mode_1 = mode_1
        self.mode_2 = mode_2
        self.check_finite = False

    def _validate_condition_metric(self, logs):
        current_1 = logs.get(self.monitor)
        current_2 = logs.get(self.monitor_2)
        
        if current_1 is None or current_2 is None:
            return False
        
        if self.best_score_1 is None :
            self.best_score_1 = current_1
            self.wait_count_1 = 0

        
        if self.best_score_2 is None :
            self.best_score_2 = current_2
            self.wait_count_2 = 0
            # return False

        cond_1 = ( self.mode_1 == "min" and (current_1 < self.best_score_1) ) or (self.mode_1 == "max" and (current_1 > self.best_score_1))
        cond_2 = ( self.mode_2 == "min" and (current_2 < self.best_score_2) ) or (self.mode_2 == "max" and (current_2 > self.best_score_2))

        if cond_1:
            self.best_score_1 = current_1
            self.wait_count_1 = 0
        else:
            self.wait_count_1 += 1

        if cond_2:
            self.best_score_2 = current_2
            self.wait_count_2 = 0
        else:
            self.wait_count_2 += 1

        return ( self.wait_count_1 >= self.patience_1) and ( self.wait_count_2 >= self.patience_2)

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if self._validate_condition_metric(logs):
            self.stopped_epoch = trainer.current_epoch
            trainer.should_stop = True
            if self.verbose:
                print(f'\nEpoch {self.stopped_epoch}: early stopping triggered.')

if __name__ == "__main__":
    parent_parser = ArgumentParser(allow_abbrev=False, add_help=False)
    # subparsers = parent_parser.add_subparsers()

    # Parse arguments
    config_trainer = PromptEngineeringLM.parse_args(parent_parser)
    config_data = DataModule.parse_args(parent_parser)
    config_model = Namespace()

    if config_trainer.test_only:
        PromptEngineeringLM.test_model(
            config_trainer, config_data, config_model)

    else:
        PromptEngineeringLM.train_model(
            config_trainer, config_data, config_model)
