# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

import json, linecache, numpy as np, pytrec_eval
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class LazyTextDataset(Dataset):
    def __init__(self, filename, include_skipped, max_seq_length, tokenizer, output_mode, load_small, dataset):
        
        self._filename = filename
        self._include_skipped = include_skipped
        self._label_list = ["False", "True"]
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._output_mode = output_mode
        self._load_small = load_small
        self._dataset = dataset
        logger.info('processing {} data'.format(dataset))
        
        self._total_data = 0
        
        if self._load_small:
            self._total_data = 100
        else:
            with open(filename, "r") as f:
                self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self._total_data
    
class ConcatModelDataset(LazyTextDataset):
    def __init__(self, filename, include_skipped, max_seq_length, tokenizer, output_mode, load_small, dataset, history_num):
        
        super(ConcatModelDataset, self).__init__(filename, include_skipped, 
                                 max_seq_length, tokenizer, output_mode, load_small, dataset)
        self._history_num = history_num

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        
        # history = [turn_1: [query, clicked_title, skipped_title],
        #            turn_2: [query, clicked_title, skipped_title],
        #                 ...
        #            turn_n-1: [query, clicked_title, skipped_title],
        #           ]
        
        data_point = json.loads(line.strip())
        guid = data_point['guid']
        query = data_point['query']
        title = data_point['title']
        label = data_point['label']
        history = data_point['history'][- self._history_num :] if self._history_num > 0 else []
        examples = []
        history_text = ''
        
        if not self._include_skipped:
            if len(history) > 0:
                history = np.asarray(history)
                history = history[:, 0:2].tolist()
        
        for history_turn in history:
            for history_behavior in history_turn:
                # if not history_behavior.startswith('[EMPTY_'):
                if history_text == '':
                    history_text += history_behavior
                else:
                    history_text += ' [SEP] ' + history_behavior

        text_a = history_text + ' [SEP] ' + query if history_text != '' else query               
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=title, label=str(label)))        
        features = convert_examples_to_features(examples, self._label_list, self._max_seq_length, 
                                   self._tokenizer, self._output_mode)
        
        input_ids = np.asarray(features[0].input_ids)
        segment_ids = np.asarray(features[0].segment_ids)
        input_mask = np.asarray(features[0].input_mask)
        ranker_label_ids = features[0].label_id
        
        # construct hier mask:
        # e.g. input_ids: [101, 2, 2, 2, 102, 3, 3, 3, 102, 4, 102, 5, 102, 6, 6, 102, 0, 0]
        #      hier_mask: [-1 , 5, 5, 5, -1 , 4, 4, 4, -1 , 3, -1 , 2, -1 , 1, 1, -1 , 0, 0]
        sep_indices = np.where(input_ids == 102)[0]
        hier_mask = np.zeros_like(input_ids)
        for sep_index in sep_indices:
            hier_mask[: sep_index + 1] += 1
        hier_mask[sep_indices] = -1
        hier_mask[0] = -1
        
        # the history mask is not used
        return {'input_ids': input_ids, 'segment_ids': segment_ids, 'input_mask': input_mask,
               'ranker_label_ids': ranker_label_ids, 'guid': guid, 'history_mask': 1, 'hier_mask': hier_mask}


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 1:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification" or output_mode == "ranking":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            # tokens_a.pop()
            # we modify this to pop from the begining since the end is always the current turn
            tokens_a.pop(0)
        else:
            # tokens_b.pop()
            tokens_b.pop(0)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels, guids=None):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == 'stateful_search':
        return trec_eval(preds, labels, guids)
    else:
        raise KeyError(task_name)
        
def trec_eval(preds, labels, guids):
    qrels = {}
    run = {}
    for pred, label, guid in zip(preds, labels, guids):
        guid_splits = guid.split('_')
        query_id = '_'.join(guid_splits[: 2])
        doc_id = guid_splits[-1]
        
        if query_id in qrels:
            qrels[query_id][doc_id] = int(label)
        else:
            qrels[query_id] = {doc_id: int(label)}
        
        if query_id in run:
            run[query_id][doc_id] = float(pred)
        else:
            run[query_id] = {doc_id: float(pred)}
            
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'ndcg'})
    res = evaluator.evaluate(run)
    mrr_list = [v['recip_rank'] for v in res.values()]
    ndcg_list = [v['ndcg'] for v in res.values()]
    return {'mrr': np.average(mrr_list), 'ndcg': np.average(ndcg_list)}, qrels, run


output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "stateful_search": "ranking",
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "stateful_search": 2,
}
