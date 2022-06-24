import torch
import numpy as np
import ast
import pandas as pd
pd.options.mode.chained_assignment = None

from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize

from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm


class Subtitle_Dataset(Dataset):
	def __init__(self, data, tokenizer, max_len):
		super().__init__()
		self.max_len = max_len
		self.data =  self.remove_nan(data)
		self.sent_tokenizer = sent_tokenize
		self.tokenizer = tokenizer
		self.len = self.data.shape[0]

	@staticmethod
	def remove_nan(data):
		data = data.replace(np.nan, '', regex=True)
		return data

	def __len__(self):
		return self.len

	def __getitem__(self,idx):
		instance = self.data.iloc[idx]

		encoded_title = self.tokenizer.encode_plus(instance['title'],max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
		title_input_ids = encoded_title['input_ids'].flatten()
		title_attention_mask = encoded_title['attention_mask'].flatten()

		encoded_subtitle = self.tokenizer.encode_plus(instance['subtitle'],max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
		subtitle_input_ids = encoded_subtitle['input_ids'].flatten()
		subtitle_attention_mask = encoded_subtitle['attention_mask'].flatten()

		sent_list = self.sent_tokenizer(instance['body'])
		encoded_body = self.tokenizer.batch_encode_plus(sent_list, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
		body_input_ids = encoded_body['input_ids']
		body_attention_mask = encoded_body['attention_mask']

		return dict(title=dict(input_ids=title_input_ids, attention_mask=title_attention_mask), 
				subtitle=dict(input_ids=subtitle_input_ids, attention_mask=subtitle_attention_mask),
				body = dict(input_ids=body_input_ids, attention_mask=body_attention_mask))


def calculate_similarity(data, tokenizer, max_len):
	'''
	This function calculate cosine similarity between title & subtitle & body
	based on bert_cosine() & tfidf_cosine() functions 
	'''	
	data['title_bert_cosine'], data['title_tfidf_cosine'], data['subtitle_bert_cosine'], data['subtitle_tfidf_cosine']  =  None, None, None, None
	tokenized_dataset = Subtitle_Dataset(data, tokenizer, max_len)
	for i in tqdm(range(data.shape[0])):
		try:
			title_bert_cosine_list, subtitle_bert_cosine_list, title_subtitle_bert_cosine_list = bert_cosine(tokenized_dataset[i]['title'], tokenized_dataset[i]['subtitle'], tokenized_dataset[i]['body'])
			title_tfidf_cosine_list, subtitle_tfdif_cosine_list = tfidf_cosine(data['title'].iloc[i], data['subtitle'].iloc[i], data['body'].iloc[i])
			assert len(sent_tokenize(data['body'].iloc[i])) == len(title_bert_cosine_list) == len(title_tfidf_cosine_list), "Check body Length" 
			assert len(title_bert_cosine_list) == len(title_tfidf_cosine_list), "Check body setences list"
			assert len(subtitle_bert_cosine_list) == len(subtitle_tfdif_cosine_list), "Check body setences list"
			data['title_bert_cosine'].iloc[i] = title_bert_cosine_list.tolist()
			data['title_tfidf_cosine'].iloc[i] = title_tfidf_cosine_list.tolist()
			data['subtitle_bert_cosine'].iloc[i] = subtitle_bert_cosine_list.tolist()
			data['subtitle_tfidf_cosine'].iloc[i] = subtitle_tfdif_cosine_list.tolist()
			data['title_subtitle_bert_cosine'].iloc[i] = title_subtitle_bert_cosine_list.tolist()
		except:
			pass
	return data


def bert_cosine(title,subtitle,body):
	'''
	calculate cosine similarity for BERT CLS Embedding

	this function returns

	* title & each body sentences BERT CLS cosine similarity
	* subtitle & each body sentences BERT CLS cosine similarity	
	* title & subtitle BERT CLS cosine similarity 	
	'''
	with torch.no_grad():
		bert_title_vec = model(input_ids = title['input_ids'].reshape(1,-1).to('cuda'), attention_mask = title['attention_mask'].reshape(1,-1).to('cuda'))
		bert_subtitle_vec = model(input_ids = subtitle['input_ids'].reshape(1,-1).cuda(), attention_mask = subtitle['attention_mask'].reshape(1,-1).cuda())
		bert_body_vec = model(input_ids = body['input_ids'].cuda(), attention_mask = body['attention_mask'].cuda())
		bert_title_vec = bert_title_vec['pooler_output'].detach().cpu().numpy()
		bert_subtitle_vec = bert_subtitle_vec['pooler_output'].detach().cpu().numpy()
		bert_body_vec = bert_body_vec['pooler_output'].detach().cpu().numpy()
	return cosine_similarity(bert_title_vec,bert_body_vec).reshape(-1), cosine_similarity(bert_subtitle_vec,bert_body_vec).reshape(-1), cosine_similarity(bert_title_vec,bert_subtitle_vec).reshape(-1)

def tfidf_cosine(title,subtitle,body):
	'''
	calculate cosine similarity for TFIDF Embedding

	this function returns

	* title & each body sentences TFIDF cosine similarity
	* subtitle & each body sentences TFIDF cosine similarity	

	'''
	tfidf_vectorizer = TfidfVectorizer(min_df=1)
	tfidf_matrix = tfidf_vectorizer.fit_transform([title]+[subtitle]+sent_tokenize(body))
	tfidf_title_vector = tfidf_matrix.toarray()[0]  # indexing target vector
	tfidf_subtitle_vector =  tfidf_matrix.toarray()[1] 
	tfidf_body_vector = tfidf_matrix.toarray()[2:]  # slicing source vectors
	return cosine_similarity(tfidf_title_vector.reshape(1,-1),tfidf_body_vector).reshape(-1), cosine_similarity(tfidf_subtitle_vector.reshape(1,-1),tfidf_body_vector).reshape(-1)

def get_pos_weight(x):
    x = np.linspace(0, x-1, x)
    return np.where(np.arctan(-(x/1.8-10))*1/1.5 > 0, np.arctan(-(x/2-10))*1/1.5, 0)

str2array = lambda x : np.array(ast.literal_eval(x)) # str list to class list
to_list = lambda x : x.tolist()

def extract_sentences(data, k, alpha, beta=None):
	'''
	extract 'k' sentences from body
	(if len(body) <= K: do not extract)

	alpha controls the weight of BERT CLS cosine similarity and TFIDF cosine similarity
	beta controls the positional weight

	ex. extract_weight =  alpha * BERT CLS cosine similarity + (1-alpha) * TFIDF cosine similarity + beta * positional weight
	'''
	data['original_body_sent_len'], data['extracted_body_sent_len'], data[f'extracted_body'] = None, None, None

	# make str similarity list to array 
	cosine_list = ['title_bert_cosine', 'title_tfidf_cosine', 'subtitle_bert_cosine', 'subtitle_tfidf_cosine']

	if type(data[cosine_list[0]].iloc[0]) == str:
		for cosine in cosine_list:
			data[cosine] = data[cosine].apply(str2array)
	else:
		pass
	
	scaler_bert = MinMaxScaler()
	scaler_tfidf = MinMaxScaler()
	for i in tqdm(range(data.shape[0])):
		title_bert_cosine_list = data['title_bert_cosine'].iloc[i]
		title_tfidf_cosine_list = data['title_tfidf_cosine'].iloc[i]
		# scale cosine similarity
		scaled_title_bert_cosine_list = scaler_bert.fit_transform(title_bert_cosine_list.reshape(-1,1)).reshape(-1)
		scaled_title_tfidf_cosine_list = scaler_tfidf.fit_transform(title_tfidf_cosine_list.reshape(-1,1)).reshape(-1) 

		if beta == None:
			# only consider BERT CLS cosine similarity and TFIDF cosine similarity
			extract_weight = alpha*(scaled_title_bert_cosine_list) + (1-alpha)*(scaled_title_tfidf_cosine_list) 
		else :
			position_weight = get_pos_weight(scaled_title_tfidf_cosine_list.shape[0])
			np.round(np.linspace(0.9499, 0.0, num=scaled_title_tfidf_cosine_list.shape[0]),1)
			assert scaled_title_bert_cosine_list.shape[0] == scaled_title_tfidf_cosine_list.shape[0] == position_weight.shape[0] , "Check weight array size"
			extract_weight = alpha*(scaled_title_bert_cosine_list) + (1-alpha)*(scaled_title_tfidf_cosine_list) + beta * position_weight
		
		body_list = sent_tokenize(data['body'].iloc[i])
		data['original_body_sent_len'].iloc[i] = len(body_list)

		# if len(body) <= k : do not extract, use whole body for subtitle generation
		if len(body_list)<=k:
				data['extracted_body'].iloc[i] = " ".join(body_list)
				data['extracted_body_sent_len'].iloc[i] = len(body_list)
		else:
			extracted_body = extract_top_k(body_list,extract_weight,k)
			data[f'extracted_body'].iloc[i] = " ".join(extracted_body)
			data['extracted_body_sent_len'].iloc[i] = extracted_body.shape[0]
	
	# numpy array > list for future applicability
	for cosine in cosine_list:
			data[cosine] = data[cosine].apply(to_list)

	return data

def extract_top_k(sent_list,extract_weight,k):
	'''
	this function returns top k sentences in positional order
	based on extract_weight

	ex. 
	k=2
	sent_list = [문장1, 문장2, 문장3, 문장4]
	extract_weight = [0.7, 0.2, 0.3, 0.1]

	return [문장1, 문장3]
	'''
	threshold = np.argsort(extract_weight)[len(sent_list)-k]
	condition = np.where(extract_weight>=extract_weight[threshold],True,False)
	return np.extract(condition,sent_list)

if __name__ == '__main__':

	tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

	print("===load model===")
	model = BertModel.from_pretrained('skt/kobert-base-v1') 
	model.eval()
	model.cuda()

	data_list = ['train','test','infer'] 
	extracted_num = 24
	for alpha in [0.9,0.8,0.7]:
		for beta in [0.1,0.2,0.3,0.4]:
			for flag in data_list:
				print(f"===load {flag} dataset===")
				data = pd.read_csv(f"dataset/{flag}.csv")

				print("====start calculating similarity====")
				data = calculate_similarity(data, tokenizer, 512)

				print(f"====extract {extracted_num} setences in body====")
				data = extract_sentences(data, extracted_num, alpha, beta)

				data.to_csv(f"dataset/{flag}_with_extracted_alpha_{alpha}_beta_{beta}.csv", index=False)
