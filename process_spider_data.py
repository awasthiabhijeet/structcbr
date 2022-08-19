import os
import sys
import json
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.data import DatasetReader, Instance
import tqdm
from allennlp.data.token_indexers import PretrainedTransformerIndexer 
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.fields import TensorField, MetadataField, TextField
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.data_loaders import SimpleDataLoader

from smbop.dataset_readers.pickle_reader import PickleReader

PRETRAINED_MODEL_PATH = "roberta-base"
TABLES_PATH = 'dataset/tables.json'
DATABASE_DIR = 'dataset/database'
PICKLE_DIR = 'processed_data'

q_token_indexer = PretrainedTransformerIndexer(model_name=PRETRAINED_MODEL_PATH)
q_tokenizer = PretrainedTransformerTokenizer(model_name=PRETRAINED_MODEL_PATH)
vocab = Vocabulary.from_pretrained_transformer(model_name=PRETRAINED_MODEL_PATH)
dataset_reader = PickleReader(
	        lazy = False,
	        question_token_indexers = {"tokens": q_token_indexer},
	        keep_if_unparsable = False,
	        tables_file = TABLES_PATH,
	        dataset_path = DATABASE_DIR,
	        cache_directory = PICKLE_DIR,
	        include_table_name_in_column=True,
	        fix_issue_16_primary_keys=False,
	        qq_max_dist=2,
	        cc_max_dist=2,
	        tt_max_dist=2,
	        max_instances=100000000,
	        decoder_timesteps=9,
	        limit_instances=-1,
	        value_pred=True,
	        use_longdb=True,)

dataset_reader.process_and_dump_pickle('dataset/dev.json', os.path.join(PICKLE_DIR,'val_original.pkl'))
dataset_reader.process_and_dump_pickle('dataset/remaining_dev.json', os.path.join(PICKLE_DIR,'val.pkl'))
dataset_reader.process_and_dump_pickle('dataset/eval_dev.json', os.path.join(PICKLE_DIR,'test.pkl'))
dataset_reader.process_and_dump_pickle('dataset/train_spider.json', os.path.join(PICKLE_DIR,'train.pkl'))
