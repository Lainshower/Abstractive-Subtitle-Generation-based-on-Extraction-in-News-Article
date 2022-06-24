import argparse
import logging
import os
from sys import flags
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset
from dataset import KoBARTSummaryDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
#from kobart import get_pytorch_kobart_model, get_kobart_tokenizer
import time

parser = argparse.ArgumentParser(description='Subtitle Generation')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default=f'/home/joonwon/subtitle_generation/extraction_generation/data/dataset/over512/new_extracted/train_with_extracted_{flag}.csv',
                            help='train file')

        parser.add_argument('--test_file',
                            type=str,
                            default=f'/home/joonwon/subtitle_generation/extraction_generation/data/dataset/over512/new_extracted/test_with_extracted_{flag}.csv',
                            help='test file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=40,
                            help='')
        parser.add_argument('--max_len',
                            type=int,
                            default=512,
                            help='max seq len')
        return parser

class KobartSummaryModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tok,
                 max_len=512,
                 batch_size=80,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        if tok is None:
            self.tok = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        else:
            self.tok = tok
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = KoBARTSummaryDataset(self.train_file_path,
                                 self.tok,
                                 self.max_len)
        self.test = KoBARTSummaryDataset(self.test_file_path,
                                self.tok,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)
        return test


class Base(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=80,
                            help='batch size for training (default: 96)')

        parser.add_argument('--lr',
                            type=float,
                            default=1e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        return optimizer


class KoBARTConditionalGeneration(Base):
    def __init__(self, hparams, **kwargs):
        super(KoBARTConditionalGeneration, self).__init__(hparams, **kwargs)
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1')
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, inputs):

        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        
        return self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KobartSummaryModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    # 어떤 alpha와 beta로 추출한 문장에 대해서 요약을 진행할 것인가?
    flag = 'alpha_0.7_beta_0.3' 

    model = KoBARTConditionalGeneration(args)
    wandb_logger = pl_loggers.WandbLogger(name=f'{flag}={args.lr}-bsz={args.batch_size}',project='subtitle_generation_over512_extracted_final', id=f'{time.strftime("%Y-%m-%d", time.localtime(time.time()))}-KoBART-extracted-{flag}-lr={args.lr}-bsz={args.batch_size}')

    dm = KobartSummaryModule(args.train_file, args.test_file, None, batch_size=args.batch_size, max_len=args.max_len, num_workers=args.num_workers)
                        
    model_chp = f'model_chp/new_extracted/{flag}'
    os.makedirs(model_chp, exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename=model_chp+'/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=2)
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, f'tb_logs/new_extracted/{flag}'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=[wandb_logger,tb_logger], accelerator='dp', callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)
