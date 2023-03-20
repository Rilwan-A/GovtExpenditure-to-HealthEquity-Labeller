#NOTE: when shuffling transformers dataset between epochs, don't forget to use flaten_indices() to ensure reads are fast

# my_iterable_dataset = my_dataset.to_iterable_dataset(num_shards=workers,flatten_indices=True)
import os, sys
sys.path.append(os.getcwd())
os.environ['TOKENIZERS_PARALLELISM'] = "false"
from sklearn.metrics import precision_recall_fscore_support
import transformers
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
from datasets import Dataset # type: ignore
import ujson as json
import yaml
from torch.utils.data import DataLoader, Dataset as TorchDataset
import glob
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.quantization import quantize_dynamic
from argparse import Namespace

from sklearn.metrics import accuracy_score

from prompt_engineering.predict import step as val_step_spot_alignment_inner
from prompt_engineering.predict import PromptBuilder, PredictionGenerator
from prompt_engineering.predict import load_dataset as load_dataset_spot_alignment
from prompt_engineering.predict import parse_args as parse_args_spot_alignment

class PromptEngineeringLM(pl.LightningModule):
    def __init__(self, 
                 model,
                 tokenizer,
                 val_task,
                 **kwargs
                 ):
        super().__init__()
        # self.nn_name = nn_name

        #NOTE: Model should be able to be loaded locally, if it is already trained
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.val_task = val_task

        if self.val_task == 'spot_alignment':
            
            self.validation_step_outputs = []

            # load in train dataset for prompt builder
            train_dset_records = load_dataset_spot_alignment('spot')[0].to_dict('records')
            
            # load in kwargs for PromptBuilder and PredictionGenerator
            spot_alignment_kwargs = parse_args_spot_alignment() 

            prompt_style = spot_alignment_kwargs.prompt_style
            k_shot = spot_alignment_kwargs.k_shot
            ensemble_size = spot_alignment_kwargs.ensemble_size
            aggregation_method = spot_alignment_kwargs.aggregation_method
            parse_output_method = spot_alignment_kwargs.parse_output_method
            
            self.prompt_builder = PromptBuilder(prompt_style, k_shot, ensemble_size, train_dset_records, self.tokenizer )
            self.prediction_generator = PredictionGenerator(self.model, self.tokenizer, prompt_style, ensemble_size, aggregation_method, parse_output_method)

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.model( input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        outputs = self(input_ids, attention_mask, labels=input_ids, output_hidden_states=False, output_attentions=False)
        
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.val_task == 'next_token':
            outp = self.validation_step_next_token(batch, batch_idx)
        elif self.val_task == 'spot_alignment':
            outp = self.validation_step_spot_alignment(batch, batch_idx)
        else:
            raise ValueError(f'val_task {self.val_task} not recognized')
        
        return outp

    def on_validation_epoch_end(self):
        
        if self.val_task == 'spot_alignment':
            self.validation_epoch_end_spot_alignment()
            self.validation_step_outputs.clear()
        else:
            pass
        return None

    def validation_step_next_token(self, batch, batch_idx):
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            outputs = self(input_ids, attention_mask, labels=input_ids, output_hidden_states=False, output_attentions=False)
        loss = outputs.loss
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def validation_step_spot_alignment(self, batch, batch_idx):

        batch_prompt_ensembles, batch_pred_ensembles, batch_pred_ensembles_parsed, batch_pred_agg = val_step_spot_alignment_inner(batch, self.prompt_builder, self.prediction_generator)

        outp = {'pred_agg': batch_pred_agg, 'label': [ d['label'] for d in batch ] } 

        self.validation_step_outputs.append(outp)

        return None
        

    def validation_epoch_end_spot_alignment(self):
        outputs = self.validation_step_outputs

        preds_agg = sum(  [ d['pred_agg'] for d in outputs ], [] )
        labels = sum( [ d['label'] for d in outputs ], [] )

        # compute metrics between binary labels and predictions
        (prec_yes, prec_no), (recall_yes, recall_no), (f1_yes, f1_no), _ = precision_recall_fscore_support(labels, preds_agg, labels=['Yes','No'], average=None)
        acc = accuracy_score(labels, preds_agg)

        self.log('val_f1/pos', f1_yes, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_prec/pos', prec_yes, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_rec/pos', recall_yes, on_epoch=True, prog_bar=True, logger=True)

        self.log('val_f1/neg', f1_no, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_prec/neg', prec_no, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_rec/neg', recall_no, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)

        self.log('val_loss', acc, on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        #optimal params for adafactor https://github.com/huggingface/transformers/pull/10526#issuecomment-804652154    
        
        # optimizer = transformers.Adafactor(self.model.parameters(), scale_parameter=True,
        #                         relative_step=True, warmup_init=True, lr=None,
        #                         weight_decay=0.01,
        #                         clip_threshold=0.5 if self.units * self.nodes >1 else 1.0
        #                         ) # Works better for small models
        # lr_scheduler = transformers.AdafactorSchedule(optimizer)

        optimizer = transformers.Adafactor(self.model.parameters(), scale_parameter=False,
                                relative_step=False, warmup_init=False, lr=1e-6,
                                weight_decay=0.01) # works better for bigger models
        lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, 
                                                                    num_training_steps=1000)
        
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}

    @staticmethod
    def train_model(config_trainer, config_data, config_model):
        
        model = transformers.AutoModelForCausalLM.from_pretrained( config_trainer.nn_name,
                                                                  load_in_8bit=False,
                                                                  device_map="auto"
                                                                  )
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(config_trainer.nn_name)
    
        # Create data module
        data_module = DataModule(**vars(config_data))

        # Create training module
        lightning_module = PromptEngineeringLM(model, tokenizer, **vars(config_trainer))
                
        #Trainer Callbacks
        callbacks = []
        callbacks.append(ModelCheckpoint(
                            monitor="val_loss",
                            filename='{epoch}-{step}-{val_loss:.3f}',
                            save_last=True,
                            auto_insert_metric_name=True,
                            save_top_k=2) )
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10))
                
        # Create trainer
        trainer = pl.Trainer(
                            strategy=config_trainer.strategy,
                            accelerator=config_trainer.accelerator,
                            devices=config_trainer.device,
                            
                            default_root_dir = os.path.join(config_trainer.dir_ckpt,f"{config_trainer.exp_name}"),

                            callbacks = callbacks,
                            precision=16,  #TODO: how does this work with load_in_8bit=True
                            
                            max_epochs=2 if config_trainer.debugging else config_trainer.max_epochs,

                            # fast_dev_run= True if config_trainer.debugging else False,
                            limit_train_batches=5 if config_trainer.debugging else None,
                            limit_val_batches=5 if config_trainer.debugging else None,
                            limit_test_batches=5 if config_trainer.debugging else None,
                            val_check_interval=config_trainer.val_check_interval )

        # Train model
        trainer.fit(lightning_module, data_module)

        # Saving configs relating to experiment
        yaml.dump( vars(config_trainer), open(os.path.join(trainer.checkpoint_callback.dirpath, 'config_trainer.yaml'),'w' ) ) #type: ignore
        yaml.dump( vars(config_data), open(os.path.join(trainer.checkpoint_callback.dirpath,'config_data.yaml'),'w' ) ) #type: ignore
        yaml.dump( vars(config_model), open(os.path.join(trainer.checkpoint_callback.dirpath,'config_model.yaml'),'w' ) )   #type: ignore

        # Get the path of the best model saved by the ModelCheckpoint Callback
        best_model_path = trainer.checkpoint_callback.best_model_path
        pt_checkpoint_path = os.path.join( best_model_path.rstrip('.ckpt')+'_pt.ckpt')
        torch.save(trainer.lightning_module.model.state_dict(), pt_checkpoint_path)

    @staticmethod
    def test_model( config_trainer, config_data, config_model=None):
        # Get paths for saved model
        dir_ckpt = config_trainer.dir_ckpt if config_trainer.dir_ckpt else ''
        dir_model = os.path.join(dir_ckpt,f"{config_trainer.exp_name}")
        dir_model_version = os.path.join(dir_model, "lightning_logs",f"version_{config_trainer.test_version}")
        
        # Load configs
        config_trainer = yaml.safe_load(open(os.path.join(dir_model_version, 'configs', 'config_trainer.yaml'), 'r'))
        config_data = yaml.safe_load( open(os.path.join(dir_model_version, 'configs', 'config_data.yaml'), 'r'))
        config_model = yaml.safe_load(open(os.path.join(dir_model_version, 'configs', 'config_model.yaml'), 'r'))

        # allowing user to update test parameters used
        changed_args_t = { k:getattr(config_trainer,k) for k in ['sample_size','batch_size_inf'] if hasattr(config_trainer,k) }
        changed_args_d = { k:getattr(config_data,k) for k in ['test_start','test_end','data_dir'] if hasattr(config_data,k) }

        for k,v in changed_args_t.items(): setattr(config_trainer,k,v)
        for k,v in changed_args_d.items(): setattr(config_data,k,v)

        # Loading parameters for saved model
        checkpoint_path = next( ( elem for elem in glob.glob(os.path.join( dir_model_version, "checkpoints", "*")) 
                                    if elem[-4:]=="ckpt"))    
            
        early_stopping_output = torch.load(checkpoint_path)['callbacks']['early_stopping']
        
        best_checkpoint_path = early_stopping_output['best_k_models'][0]['path']

        # Load a transformer model from checkpoint path using a pytorch lightning checkpoint
        pt_checkpoint_path = os.path.join( best_checkpoint_path.rstrip('.ckpt')+'_pt.ckpt' , '.ckpt' )
        model = transformers.AutoModelForCausalLM.from_pretrained( pt_checkpoint_path ,
                                                                  load_in_8bit=True,
                                                                  device_map="auto")

        tokenizer = transformers.AutoTokenizer.from_pretrained(config_trainer.nn_name)
        
        # Making training module
        lightning_module = PromptEngineeringLM(model, tokenizer, **vars(config_trainer))

        # Making DataModule
        data_module = DataModule(**vars(config_data))

        trainer = pl.Trainer(  
                    accelerator=config_trainer.accelerator,
                    devices=config_trainer.device,
                    strategy=config_trainer.strategy,
                    default_root_dir = dir_model_version,
                    logger=False,

                    # precision=16,
                    # native_amp=True
                    )

        
        # Test the Trainer
        trainer.test(
            lightning_module,
            ckpt_path=checkpoint_path,
            dataloaders=data_module.test_dataloader())
        

    @staticmethod
    def parse_args(parent_parser=None):
        
        parser = ArgumentParser(parents=[parent_parser] if parent_parser else None, add_help=True, allow_abbrev=False)

        # parser = subparsers.add_parser('trainer', add_help=True, allow_abbrev=False)

        parser.add_argument('--exp_name',type=str, default='debug')
        parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B' )
        
        # parser.add_argument('--dir_ckpt',type=str, required=True)
        parser.add_argument('--dir_ckpt',type=str, default='prompt_engineering/finetune_checkpoints', required=True )
                
        parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp','deepspeed'])
        parser.add_argument('--accelerator', type=str, default='gpu')
        parser.add_argument('--device', type=int, default=1)
        
        parser.add_argument('--max_epochs', type=int, default=50)
        parser.add_argument('--val_check_interval', type=int,  default=1.0 )

        # let user pass in a multiple values for a argument called early_stopping_metrics
        parser.add_argument('--test_only', action='store_true', help='Include flag to test only')
        parser.add_argument('--debugging', action='store_true')

        parser.add_argument('--val_task', type=str, choices=['next_token','spot_alignment'], default='next_token')

        args = parser.parse_known_args()[0]

        return args
    
class DataModule(pl.LightningDataModule):

    def __init__(self, nn_name, dir_data, batch_size, batch_size_inf, val_task, num_workers=4):
        super().__init__()
        self.nn_name = nn_name
        self.dir_data = dir_data
        self.num_workers = num_workers

        self.batch_size = batch_size
        self.batch_size_inf = batch_size_inf

        self.val_task = val_task
    
    def train_dataloader(self):
        
        dataset = Dataset.load_from_disk( os.path.join(self.dir_data, f"{self.nn_name.replace('/','_')}/train.arrow") )
        # dataset = dataset.to_iterable_dataset(num_shards=self.num_workers*2)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return dataloader

    def val_dataloader(self):
        if self.val_task == 'next_token':
            dataset = Dataset.load_from_disk( 
                os.path.join(self.dir_data, f"{self.nn_name.replace('/','_')}/val.arrow") ).with_format("torch")
            
            # dataset = dataset.to_dataset() to_iterable_dataset(num_shards=self.num_workers*2)
            dataloader = DataLoader(dataset, batch_size=self.batch_size_inf,
                                    shuffle=True, num_workers=self.num_workers, pin_memory=True)
            
        elif self.val_task == 'spot_alignment':
            test_records = load_dataset_spot_alignment('spot')[1].to_dict('records')
                      
            # initialize pytorch dataset from records
            test_dset = SpotDataset( test_records )

            dataloader = DataLoader(test_dset, batch_size=self.batch_size_inf,
                                    shuffle=False,
                                    num_workers=self.num_workers,
                                    pin_memory=False,
                                    collate_fn=SpotDataset.collate_fn,
                                    drop_last=False)
        else:
            raise ValueError(f"val_task {self.val_task} not recognized")
            
        return dataloader

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError
        return super().test_dataloader()

    @staticmethod
    def parse_args(parent_parser=None):
        parser = ArgumentParser(parents=[parent_parser] if parent_parser else None, add_help=True, allow_abbrev=False)

        # parser = subparsers.add_parser('data', add_help=True, allow_abbrev=False)

        parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B' )
        parser.add_argument('--dir_data',type=str, default='./datasets/finetune/preprocessed/')

        parser.add_argument('--batch_size',type=int, default=1)
        parser.add_argument('--batch_size_inf',type=int, default=2)

        parser.add_argument('--val_task', type=str, choices=['next_token','spot_alignment'], default='next_token')
               
        args = parser.parse_known_args()[0]
        return args

class SpotDataset(TorchDataset):
    def __init__(self, data):
        self.data = data

        for idx in range(len(self.data)):
            self.data[idx].pop('id')
            self.data[idx].pop('type')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Get an item from the data list
        item = self.data[index]       

        return item

    @staticmethod
    def collate_fn(batch):
        return batch

if __name__ == "__main__":
    parent_parser = ArgumentParser( allow_abbrev=False, add_help=False)
    # subparsers = parent_parser.add_subparsers()

    # Parse arguments
    config_trainer = PromptEngineeringLM.parse_args(parent_parser)
    config_data = DataModule.parse_args(parent_parser)
    config_model = Namespace()
    
    if config_trainer.test_only:
        PromptEngineeringLM.test_model(config_trainer, config_data, config_model)

    else:
        PromptEngineeringLM.train_model(config_trainer, config_data, config_model)