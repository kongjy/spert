import argparse
import math
import os
from typing import Type

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

import sys
sys.path.append("..") 
from spert import models, prediction
from spert import sampling
from spert import util
from spert.entities import Dataset
from spert.evaluator import Evaluator
from spert.input_reader import BaseInputReader
from spert.loss import SpERTLoss, Loss
from tqdm import tqdm
from spert.trainer import BaseTrainer

from spert import input_reader
from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs

import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer

from spert import util
from spert.entities import Dataset, EntityType, RelationType, Entity, Relation, Document
from spert.opt import spacy


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 neg_rel_count: int = None, max_span_size: int = None):
        super().__init__(types_path, tokenizer, neg_entity_count, neg_rel_count, max_span_size)

    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._relation_types, self._entity_types, self._neg_entity_count,
                          self._neg_rel_count, self._max_span_size)
        #print('dataset is:', dataset)
        self._parse_dataset(dataset_path, dataset)
        #self._datasets[dataset_label] = dataset
    
    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        counter = 0 
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            print('document #:', counter)
            counter += 1
            self._parse_document(document, dataset)
    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['tokens']
        jrelations = doc['relations']
        jentities = doc['entities']
        #print('jokens length:', len(jtokens))
        #print('jrelations length:', len(jrelations)) 
        #print('jentities length:', len(jentities))

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)
        #print('parsed tokens and entities, gonna parse relations.') 
 
        # parse relations
        #relations = self._parse_relations(jrelations, entities, dataset)
        relations = self._parse_relations(jrelations, jentities, jtokens, entities, dataset)
        # create document
        document = dataset.create_document(doc_tokens, entities, relations, doc_encoding)
        return document

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']
            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    #def _parse_relations(self, jrelations, entities, dataset) -> List[Relation]:
    def _parse_relations(self, jrelations, jentities, jtokens, entities, dataset) -> List[Relation]:   
        relations = []
        #print('jrelations', jrelations)
        #print(dir(entities[0]))
        #for i in range(len(entities)):
        #    print(entities[i])
        #    print(entities[i].phrase)
        #    print(type(entities[i].phrase))
        #    print(len(entities[i].phrase)) 
        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['type']]

            head_idx = jrelation['head']
            tail_idx = jrelation['tail']

            # create relation
            #print('head_idx:', head_idx)
            #print('tail_idx:', tail_idx)
            head_j = jtokens[jentities[jrelation['head']]['start']]
            tail_j = jtokens[jentities[jrelation['tail']]['start']]
            head = entities[head_idx]
            tail = entities[tail_idx]
            if head_j != entities[head_idx].phrase:
                #print('Not a match for head:')
                #print('---------------------')
                #print(head, 'vs.', head_j)
                if len(entities[head_idx].phrase) == 0 or entities[head_idx].phrase == None:
                    
                    print('entities length:', len(entities)) 
                    print('head_idx:', head_idx) 
                    print('tail_idx:', tail_idx) 
                    print(head, 'vs.', head_j) 
            if tail_j != entities[tail_idx].phrase:
                #print('Not a match for tail:')
                #print('---------------------')
                #print(tail, 'vs.', tail_j)
                if len(entities[tail_idx].phrase) == 0 or entities[tail_idx].phrase == None:  
                    print(tail, 'vs.', tail_j)
            if len(head.phrase) and len(tail.phrase) >0:
                reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)

                # for symmetric relations: head occurs before tail in sentence
                if relation_type.symmetric and reverse:
                    head, tail = util.swap(head, tail)
        
                relation = dataset.create_relation(relation_type, head_entity=head, tail_entity=tail, reverse=reverse)
                relations.append(relation)

        return relations


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)

def _parse_tokens(jtokens, dataset, tokenizer):
    doc_tokens = []

    # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

    # parse tokens
    for i, token_phrase in enumerate(jtokens):
        token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)
        if not token_encoding:
            token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
        span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

        token = dataset.create_token(i, span_start, span_end, token_phrase)

        doc_tokens.append(token)
        doc_encoding += token_encoding

    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]

    return doc_tokens, doc_encoding
class SpERTTrainer(BaseTrainer):
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # byte-pair encoding
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)

    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # read datasets
        #input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
        #                                args.neg_relation_count, args.max_span_size, self._logger)
        input_reader = JsonInputReader(types_path, self._tokenizer, args.neg_entity_count,
                                       args.neg_relation_count, args.max_span_size)
        train_dataset = input_reader.read(train_path, train_label)
        print('train_dataset is of type', type(train_dataset))
        validation_dataset = input_reader.read(valid_path, valid_label)
        print('validation dataset is of type', type(validation_dataset)) 
        print('validation dataset:', validation_dataset)

if __name__ == "__main__": 
    arg_parser = argparse.ArgumentParser() 
    args, _ = arg_parser.parse_known_args()
    arg_parser = train_argparser()  
    process_configs(target=__train, arg_parser=arg_parser)
