#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from tqdm import tqdm
import numpy as np


# In[2]:


# preds = '/mnt/scratch/chenqu/stateful_search/29_eval/test_preds.txt'
preds = '/mnt/scratch/chenqu/stateful_search/test_preds.txt'
dev_file = '/mnt/scratch/chenqu/msmarco/preprocessed/session_dev_small.txt'
test_file = '/mnt/scratch/chenqu/msmarco/preprocessed/session_test.txt'
train_file = '/mnt/scratch/chenqu/msmarco/preprocessed/session_train_original.txt'


# In[3]:


with open(preds) as fin:
    res = json.load(fin)


# In[4]:


res.keys()


# In[8]:


def mrr(preds, labels, doc_num_list):
    mrr_list = []
    for num in tqdm(doc_num_list):
        cur_preds = preds[: num]
        cur_labels = labels[: num]
        mrr_list.append(single_mrr(cur_preds, cur_labels))
        
        preds = preds[num :]
        labels = labels[num :]
        
    return {'mrr': np.average(mrr_list)}, mrr_list
        
def single_mrr(preds, labels):
    score = 0.0
    index_rank = np.argsort(preds)[::-1]
    for rank, i in enumerate(index_rank):
        if labels[i] == 1:
            score = 1.0 / (rank + 1.0)
            break
            
    return score


# In[9]:


print(mrr(res['preds'], res['ranker_test_all_label_ids'], res['test_doc_num_list']))


# In[17]:


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


# In[ ]:


guids = []
qid, did = 0, 0
for num in tqdm(res['test_doc_num_list']):
    qid += 1
    did = 0
    for i in range(num):
        did += 1
        guids.append('fake_{}_{}'.format(qid, did))


# In[19]:


import pytrec_eval
trec_eval(res['preds'], res['ranker_test_all_label_ids'], guids)


# In[ ]:




