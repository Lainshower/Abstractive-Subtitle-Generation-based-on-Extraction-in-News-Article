# pytorch-lightning binary --> huggingface binary로 추출 작업 필요

import argparse
from train import KoBARTConditionalGeneration
from transformers.models.bart import BartForConditionalGeneration
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument("--hparams", default=None, type=str) # logs/tblogs/configuration/default/version_0/hparams.yaml
parser.add_argument("--model_binary", default=None, type=str) #logs/model_chp/~.ckpt
parser.add_argument("--output_dir", default=None, type=str)
args = parser.parse_args()

with open(args.hparams) as f:
    hparams = yaml.load(f)
    inf = KoBARTConditionalGeneration.load_from_checkpoint(args.model_binary, hparams=hparams)
    inf.model.save_pretrained(args.output_dir)
