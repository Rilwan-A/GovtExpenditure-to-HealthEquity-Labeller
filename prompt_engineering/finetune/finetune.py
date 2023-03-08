#NOTE: when shuffling transformers dataset between epochs, don't forget to use flaten_indices() to ensure reads are fast

# my_iterable_dataset = my_dataset.to_iterable_dataset(num_shards=workers,flatten_indices=True)
import os
import transformers
import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
from datasets import Dataset

class Trainer(pl.LightningModule):
    def __init__(self, nn_name):
        super().__init__()
        self.nn_name = nn_name

        #NOTE: Model should be able to be loaded locally, if it is already trained
        self.model = transformers.AutoModel.from_pretrained( nn_name ,load_in_8bit=True, device_map="auto")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(nn_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        outputs = self(input_ids, attention_mask)
        
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step_spot_alignment(self, batch, batch_idx):
        return None

    def configure_optimizers(self):
        #optimal params for adafactor https://github.com/huggingface/transformers/pull/10526#issuecomment-804652154    
        
        # optimizer = transformers.Adafactor(self.model.parameters(), scale_parameter=True,
        #                         relative_step=True, warmup_init=True, lr=None,
        #                         weight_decay=0.005,
        #                         clip_threshold=0.5 if self.units * self.nodes >1 else 1.0
        #                         ) # Works better for small models

        optimizer = transformers.Adafactor(self.model.parameters(), scale_parameter=False,
                                relative_step=False, warmup_init=False, lr=1e-3,
                                weight_decay=0.01) # works better for bigger models

        lr_scheduler = transformers.AdafactorSchedule(optimizer)
        
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_loss'}


    @staticmethod
    def start(trainer, tparams, training_module, data_module, mparams):

        if tparams['mode'] in ['train_new', 'train_cont']:
            trainer.fit(training_module)

        if tparams['mode'] in ["test"]:

            checkpoint = Trainer.get_ckpt_file(tparams['dir_checkpoints'])
            training_module.load_state_dict(checkpoint['state_dict'])

            training_module.eval()
            training_module.freeze()

            dict_results = trainer.test(
                test_dataloaders=data_module.test_dataloader(),
                model=training_module)

            # Saving test results for model to file
            _dir = os.path.join(tparams['model_dir'], mparams['model_name'])
            fn = os.path.join(_dir, "results.json")

            if os.path.isfile(fn) == False:
                existing_results = {}
            else:
                with open(fn, 'r') as outfile:
                    existing_results = json.load(outfile)

            existing_results[f"{mparams['model_name']}_{tparams['version']}"] = dict_results[0]['test_loss']

            with open(fn, 'w') as outfile:
                json.dump(existing_results, outfile)

    @staticmethod
    def get_ckpt_file(_dir_checkpoint, mode='best'):
        if mode == 'best':
            checkpoint_yaml_file = os.path.join(_dir_checkpoint, "best_k_models.yaml")
            
            scores_dict = yaml.load(open(checkpoint_yaml_file, "r"), Loader=yaml.FullLoader)
            best_ckpt_path = min(scores_dict, key=scores_dict.get)

            if os.path.exists(best_ckpt_path) == False:
                root_dir = Path(__file__).resolve().parents[4]
                best_ckpt_path = os.path.join(
                    str(root_dir), best_ckpt_path[best_ckpt_path.index('mastering-conversation'):])

            checkpoint = torch.load(best_ckpt_path, map_location='cpu')

        else:
            raise NotImplementedError

        return checkpoint
        


def train_dataloader(self,
                        nn_name:str,
                        data_dir:str,
                        num_workers=4):
    
    dataset = Dataset.load_from_disk( os.path.join(data_dir, f"./datasets/finetune/preprocessed/{nn_name.replace('/','_')}/train.arrow") )
    dataset = dataset.to_iterable_dataset(num_shards=num_workers*2,flatten_indices=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True, reload_dataloader_every_epoch=True)
    return dataloader

def val_dataloader(self,
                        nn_name:str,
                        data_dir:str,
                        num_workers=4):
    
    dataset = Dataset.load_from_disk( os.path.join(data_dir, f"./datasets/finetune/preprocessed/{nn_name.replace('/','_')}/val.arrow") )
    dataset = dataset.to_iterable_dataset(num_shards=num_workers*2,flatten_indices=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, pin_memory=True, reload_dataloader_every_epoch=True)
    return dataloader





def parse_args():
    
    parser = ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument('--nn_name', type=str, default='EleutherAI/gpt-j-6B' )
    parser.add_argument('--data_dir',type=str, default='datasets/finetune/preprocessed/' )
    parser.add_argument('--min_word_per_chunk',type=int, default=10)

    args = parser.parse_known_args()[0]

    return args


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    main(**vars(args))