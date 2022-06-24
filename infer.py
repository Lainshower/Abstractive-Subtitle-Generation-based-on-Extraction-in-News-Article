'''
Fine-tuning한 KoBART checkpoint에서 모델 가져와서 Inference하는 코드
'''

import torch
import numpy as np
from tqdm import *
import pandas as pd
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

# Load Model for analysis
model = BartForConditionalGeneration.from_pretrained("model_checkpoint_pytorch/")
model.cuda()
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')

# Load Infered Data
infer_data  = pd.read_csv("dataset/infer_similarity_calculated.csv")

max_len = 512

# Infering Subtitle 
infer_data['generated_subtitle'] = np.nan
infer_data = infer_data[-infer_data.body.isna()] # 추출한 본문이 결측치로 되어있는 경우에는 제거하고 실험
infer_data.reset_index(drop=True, inplace=True)

for i in tqdm(range(len(infer_data))):
	'''
	print(i)
	print('Title : ',infer_data['title'][i])
	print('Body : ', infer_data['body'][i])
	input_ids = tokenizer.encode(infer_data['body'][i])
	input_ids = torch.tensor(input_ids)
	input_ids = input_ids.unsqueeze(0)
	output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
	output = tokenizer.decode(output[0], skip_special_tokens=True)
	print('Reference Subtitle : ', infer_data['subtitle'][i])
	print('Generated Subtitle : ', output)
	'''
	try: 
		input_ids = tokenizer.encode(infer_data['body'].iloc[i])
		input_ids = input_ids[:max_len]
		input_ids = torch.tensor(input_ids)
		input_ids = input_ids.unsqueeze(0)
		output = model.generate(input_ids.to('cuda'), eos_token_id=1, max_length=128, num_beams=5)
		output.detach().cpu()
		output = tokenizer.decode(output[0], skip_special_tokens=True)
		print(output)
		infer_data['generated_subtitle'][i] = output
	except:
		pass


infer_data.to_csv("dataset/infer_done.csv", index=False)
