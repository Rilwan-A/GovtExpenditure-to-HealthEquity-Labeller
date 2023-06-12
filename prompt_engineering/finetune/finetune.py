import os
import sys
sys.path.append(os.getcwd())
os.environ['TOKENIZERS_PARALLELISM'] = "false"

import deepspeed
import bitsandbytes as bnb

from sklearn.metrics import accuracy_score
from argparse import Namespace
from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import glob
from torch.utils.data import DataLoader, Dataset as TorchDataset
import yaml
import ujson as json
from datasets import Dataset  # type: ignore
import torch
from argparse import ArgumentParser
from lightning.pytorch import loggers as pl_loggers
import lightning.pytorch as pl
import transformers
from sklearn.metrics import precision_recall_fscore_support

from prompt_engineering.huggingface.predict import parse_args as parse_args_spot_alignment
from prompt_engineering.huggingface.predict import load_dataset as load_dataset_spot_alignment
from prompt_engineering.huggingface.predict import PromptBuilder, PredictionGenerator
from prompt_engineering.huggingface.predict import step as val_step_spot_alignment_inner

# We also advise users to use the nested quantization technique. This saves more memory at no additional performance - 
# from our empirical observations, this enables fine-tuning llama-13b model on an NVIDIA-T4 16GB with a sequence length of 1024,
#  batch size of 1 and gradient accumulation steps of 4.

# from lightning.pytorch.plugins.training_type import DeepSpeedPlugin
class PromptEngineeringLM(pl.LightningModule):
    """LightningModule for prompt engineering LM training and evaluation."""
    def __init__(self,
                 model,
                 tokenizer,
                 val_task,
                 **kwargs
                 ):
        super().__init__()
        # self.nn_name = nn_name

        # NOTE: Model should be able to be loaded locally, if it is already trained
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.val_task = val_task
        self.optimizer = kwargs.get('optimizer', 'auto')
        self.lr = kwargs.get('lr', 1e-6)

        if self.val_task == 'spot_alignment':

            self.validation_step_outputs = []

            # load in train dataset for prompt builder
            train_dset_records = load_dataset_spot_alignment('spot')[
                0].to_dict('records')

            # load in kwargs for PromptBuilder and PredictionGenerator
            spot_alignment_kwargs = parse_args_spot_alignment()

            prompt_style = spot_alignment_kwargs.prompt_style
            k_shot = spot_alignment_kwargs.k_shot
            ensemble_size = spot_alignment_kwargs.ensemble_size
            aggregation_method = spot_alignment_kwargs.aggregation_method
            parse_output_method = spot_alignment_kwargs.parse_output_method

            self.prompt_builder = PromptBuilder(
                prompt_style, k_shot, ensemble_size, train_dset_records, 'indirectly', self.tokenizer)
            self.prediction_generator = PredictionGenerator(
                self.model, self.tokenizer, prompt_style, ensemble_size, aggregation_method, parse_output_method, deepspeed_compat=True)

    def freeze_layers_fn(self):
        for idx, layer in enumerate(self.model.transformer.h):
            if idx < self.freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        outputs = self(input_ids, attention_mask, labels=input_ids,
                       output_hidden_states=False, output_attentions=False)

        loss = outputs.loss
        self.log('train_loss', loss, on_step=True,
                 on_epoch=False, prog_bar=True, logger=True,
                sync_dist=False, rank_zero_only=True)
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
    
    def validation_step_next_token(self, batch, batch_idx):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            outputs = self(input_ids, attention_mask, labels=input_ids,
                           output_hidden_states=False, output_attentions=False)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step_spot_alignment(self, batch, batch_idx):

        batch_prompt_ensembles, batch_pred_ensembles, batch_pred_ensembles_parsed, batch_pred_agg = val_step_spot_alignment_inner(
            batch, self.prompt_builder, self.prediction_generator)

        outp = {'pred_agg': batch_pred_agg,
                'label': [d['label'] for d in batch]}

        self.validation_step_outputs.append(outp)

        return None

    def validation_epoch_end_spot_alignment(self):          
        
        outputs = self.validation_step_outputs

        preds_agg = sum([d['pred_agg'] for d in outputs], [])
        labels = sum([d['label'] for d in outputs], [])

        # compute metrics between binary labels and predictions
        (prec_yes, prec_no), (recall_yes, recall_no), (f1_yes, f1_no), _ = precision_recall_fscore_support(labels, preds_agg, labels=['Yes', 'No'], average=None)
        
        acc = accuracy_score(labels, preds_agg)

        self.log('val_loss', acc, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)
        
        self.log('val_acc', acc, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)
        
        self.log('val_f1/pos', f1_yes, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)
        self.log('val_prec/pos', prec_yes, on_step=False, on_epoch=True,
                prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        self.log('val_rec/pos', recall_yes, on_step=False, on_epoch=True,
                prog_bar=False, logger=True,  rank_zero_only=False, sync_dist=True)

        self.log('val_f1/neg', f1_no, on_step=False, on_epoch=True,
                prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)
        self.log('val_prec/neg', prec_no, on_step=False, on_epoch=True,
                prog_bar=False, logger=True,  rank_zero_only=False, sync_dist=True)
        self.log('val_rec/neg', recall_no, on_step=False, on_epoch=True,
                prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)

    def configure_optimizers(self):
        # optimal params for Adafactor https://github.com/huggingface/transformers/pull/10526#issuecomment-804652154

        if self.optimizer == 'Adafactor':
            optimizer = transformers.Adafactor(self.trainer.model.model.parameters(), scale_parameter=False,
                                               relative_step=False, warmup_init=False, lr=self.lr,
                                               weight_decay=0.01)  # works better for bigger models
            lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200,
                                                                        num_training_steps=1000)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}

        elif self.optimizer == 'Adam8bit':
            optimizer = bnb.optim.Adam8bit(
                    self.trainer.model.model.parameters(), 
                    lr=self.lr, betas=(0.9, 0.995)
                    )
            return {'optimizer': optimizer, 'monitor': 'val_loss'}

        elif self.optimizer == 'DeepSpeedCPUAdam':
            # crate a Adam optimizer from deepspeed library 
            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.trainer.model.model.parameters(),
                                                             lr=self.lr, betas=(0.9, 0.995),
                                                             fp32_optimizer_states=False)
            return {'optimizer': optimizer, 'monitor': 'val_loss'}

        elif self.optimizer == 'Adam':
            # crate a Adam optimizer from deepspeed library 
            optimizer = deepspeed.ops.adam.FusedAdam(self.trainer.model.model.parameters(), 
                                                     lr=self.lr,
                                                     betas=(0.9, 0.995))
            return {'optimizer': optimizer, 'monitor': 'val_loss'}

        else:
            raise ValueError(f'optimizer {self.optimizer} not recognized')

    @staticmethod
    def train_model(config_trainer, config_data, config_model):

        model = transformers.AutoModelForCausalLM.from_pretrained(config_trainer.nn_name,
                                                                  load_in_8bit=False)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config_trainer.nn_name)

        # Create data module
        data_module = DataModule(**vars(config_data))

        # Create training module
        lightning_module = PromptEngineeringLM(
            model, tokenizer, **vars(config_trainer))

        # Trainer Callbacks
        callbacks = []
        callbacks.append(ModelCheckpoint(
            monitor="val_loss",
            filename='{epoch}-{step}-{val_loss:.3f}',
            save_last=False,
            auto_insert_metric_name=True,
            save_weights_only=True,
            save_top_k=1))
        callbacks.append(EarlyStopping(monitor="val_loss", patience=8))

        # setting up strategy
        strategy = config_trainer.strategy
        ds_config = {
                    # "optimizer": {
                    #     "type": "AdamW",
                    #     "params": {
                    #         "lr": "auto",
                    #         "betas": "auto",
                    #         "eps": "auto",
                    #         "weight_decay": "auto"
                    #            }
                    #        },
                    # "scheduler": {
                    #     "type": "WarmupLR",
                    #     "params": {
                    #         "last_batch_iteration": -1,
                    #         "warmup_min_lr": 0,
                    #         "warmup_max_lr": 3e-5,
                    #         "warmup_num_steps": 100,
                    #         },
                    #     },
                    "zero_optimization": {
                        "stage": 3,
                        "offload_optimizer": {
                            "device": "cpu",
                            "pin_memory": True
                        },
                        "offload_param": {
                            "device": "cpu",
                            "pin_memory": True
                        },

                        "overlap_comm": True,
                        "contiguous_gradients": True,
                        "sub_group_size": 1e9,
                        "reduce_bucket_size": 2e8,
                        "stage3_prefetch_bucket_size": "auto",
                        "stage3_param_persistence_threshold": "auto",
                        "stage3_max_live_parameters": 1e9,
                        "stage3_max_reuse_distance": 1e9,
                        "stage3_gather_16bit_weights_on_model_save": False
                    }
                }
        strategy = DeepSpeedStrategy(config=ds_config)

        # Create trainer
        trainer = pl.Trainer(
            strategy=strategy,

            accelerator=config_trainer.accelerator,
            devices=config_trainer.devices,

            default_root_dir=os.path.join(
                config_trainer.dir_ckpt, f"{config_trainer.exp_name}"),

            callbacks=callbacks,
            logger=pl_loggers.TensorBoardLogger(
                save_dir=config_trainer.dir_ckpt),
            
            # precision= '16-mixed',
            precision= 'bf16',
            accumulate_grad_batches=config_trainer.accumulate_grad_batches,

            max_epochs=2 if config_trainer.debugging else config_trainer.2e-5	,
            num_sanity_val_steps=4,
            
            limit_train_batches=2 if config_trainer.debugging else None,
            limit_val_batches=1 if config_trainer.debugging else None,
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

        # Only run on main thread
        if trainer.global_rank == 0:
            from deepspeed.utils import zero_to_fp32
            import shutil
            # Run the to_fp32.py script to convert the model to fp32
            best_model_dir = trainer.checkpoint_callback.best_model_path
            output_file = os.path.join(os.path.dirname(best_model_dir), 'pytorch_model.bin')
            zero_to_fp32.convert_zero_checkpoint_to_fp32_state_dict(best_model_dir, output_file)
                    
            # Then Delete the ckpt files saved by DeepSpeed to save memory
            shutil.rmtree(os.path.join(best_model_dir), ignore_errors=True)

            # Clear Trash Bin on linux computer
            os.system('rm -rf ~/.local/share/Trash/*')
    
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
            'sample_size', 'batch_size_inf'] if hasattr(config_trainer, k)}
        changed_args_d = {k: getattr(config_data, k) for k in [
            'test_start', 'test_end', 'data_dir'] if hasattr(config_data, k)}

        for k, v in changed_args_t.items():
            setattr(config_trainer, k, v)
        for k, v in changed_args_d.items():
            setattr(config_data, k, v)

        # Loading parameters for saved model
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
            config_trainer.nn_name)

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

        parser.add_argument('--exp_name', type=str, default='debug')
        parser.add_argument('--nn_name', type=str,
                            default='EleutherAI/gpt-j-6B')
        parser.add_argument('--freeze_layers',type=int, default=0)

        parser.add_argument(
            '--dir_ckpt', type=str, default='prompt_engineering/finetune/ckpt', required=True)

        parser.add_argument('--strategy', type=str, default='auto',
                            choices=['auto', 'ddp', 'fsdp', 'deepspeed_stage_2', 'deepspeed_stage_2_offload',
                                     'deepspeed_stage_3', 'deepspeed_stage_3_offload'])

        parser.add_argument('--optimizer', type=str,
                            choices=['Adafactor', 'OneBitAdam', 'DeepSpeedCPUAdam', 'Adam8bit', 'Adam', 'Adamw' ,'ZeroOneAdam'], 
                            default='Adafactor')
        parser.add_argument('--lr', type=float, default=1e-5)
        parser.add_argument('--accelerator', type=str, default='auto')
        parser.add_argument('--devices', type=int, default=1)
        parser.add_argument('--accumulate_grad_batches', type=int, default=1)

        parser.add_argument('--max_epochs', type=int, default=3)
        parser.add_argument('--val_check_interval', type=int,  default=1.0)

        # let user pass in a multiple values for a argument called early_stopping_metrics
        parser.add_argument('--test_only', action='store_true',
                            help='Include flag to test only')
        
        parser.add_argument('--debugging', action='store_true')

        parser.add_argument(
            '--val_task', type=str, choices=['next_token', 'spot_alignment'], default='next_token')

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

        dataset = Dataset.load_from_disk(os.path.join(
            self.dir_data, f"{self.nn_name.replace('/','_')}/train.arrow"))
        # dataset = dataset.to_iterable_dataset(num_shards=self.num_workers*2)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=self.num_workers, 
                                pin_memory=False, drop_last=False
                                )
        return dataloader

    def val_dataloader(self):
        if self.val_task == 'next_token':
            dataset = Dataset.load_from_disk(
                os.path.join(self.dir_data, f"{self.nn_name.replace('/','_')}/val.arrow")).with_format("torch")

            # dataset = dataset.to_dataset() to_iterable_dataset(num_shards=self.num_workers*2)
            dataloader = DataLoader(dataset, batch_size=self.batch_size_inf,
                                    shuffle=False, num_workers=self.num_workers,
                                    pin_memory=False)

        elif self.val_task == 'spot_alignment':
            test_records = load_dataset_spot_alignment(
                'spot')[1].to_dict('records')

            # initialize pytorch dataset from records
            test_dset = SpotDataset(test_records)

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

        
        parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B')
        parser.add_argument('--dir_data', type=str, default='./data/finetune/preprocessed/')

        parser.add_argument('--num_workers', type=int, default=0)

        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--batch_size_inf', type=int, default=2)

        parser.add_argument('--val_task', type=str, choices=['next_token', 'spot_alignment'], default='next_token')

        args = parser.parse_known_args()[0]
        return args

class SpotDataset(TorchDataset):
    def __init__(self, data):
        self.data = data

        for idx in range(len(self.data)):
            _ = self.data[idx].pop('id')
            _ = self.data[idx].pop('type')

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
