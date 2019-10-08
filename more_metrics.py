#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from tqdm import tqdm
import numpy as np
import pytrec_eval


# In[2]:


preds = '/mnt/scratch/chenqu/stateful_search/97/test_preds.txt'
dev_file = '/mnt/scratch/chenqu/aol/preprocessed/session_dev_small.txt'
test_file = '/mnt/scratch/chenqu/aol/preprocessed/session_test.txt'
train_file = '/mnt/scratch/chenqu/aol/preprocessed/session_train.txt'


# In[3]:


with open(preds) as fin:
    res = json.load(fin)
qrels = res['qrels']
run = res['run']


# In[4]:


evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'ndcg', 'ndcg_cut'})
eval_res = evaluator.evaluate(run)
mrr_list = [v['recip_rank'] for v in eval_res.values()]
ndcg_list = [v['ndcg'] for v in eval_res.values()]
ndcg_1_list = [v['ndcg_cut_1'] for v in eval_res.values()]
ndcg_3_list = [v['ndcg_cut_3'] for v in eval_res.values()]
ndcg_10_list = [v['ndcg_cut_10'] for v in eval_res.values()]

print('mrr', np.average(mrr_list))
print('ndcg_list', np.average(ndcg_list))
print('ndcg_1_list', np.average(ndcg_1_list))
print('ndcg_3_list', np.average(ndcg_3_list))
print('ndcg_10_list', np.average(ndcg_10_list))


# In[10]:


# train_dict = {}
# train_guid_dict = {}
# with open(train_file) as fin:
#     for line in tqdm(fin):
#         dp = json.loads(line)
#         train_dict[(dp['query'], dp['title'], dp['label'])] = 1
#         # train_guid_dict[dp['guid']] = 1
        
# counter = 0
# with open(test_file) as fin:
#     for line in tqdm(fin):
#         dp = json.loads(line)
#         if (dp['query'], dp['title'], dp['label']) in train_dict:
#         # if dp['guid'] in train_guid_dict:
#             counter += 1
# print(counter)


# In[11]:


# res.keys()


# In[5]:


# def mrr(preds, labels, doc_num_list):
#     mrr_list = []
#     for num in tqdm(doc_num_list):
#         cur_preds = preds[: num]
#         cur_labels = labels[: num]
#         mrr_list.append(single_mrr(cur_preds, cur_labels))
        
#         preds = preds[num :]
#         labels = labels[num :]
        
#     return {'mrr': np.average(mrr_list)}, mrr_list
        
# def single_mrr(preds, labels):
#     score = 0.0
#     index_rank = np.argsort(preds)[::-1]
#     for rank, i in enumerate(index_rank):
#         if labels[i] == 1:
#             score = 1.0 / (rank + 1.0)
#             break
            
#     return score


# In[6]:


# mrr(res['preds'], res['ranker_test_all_label_ids'], [50] * len(qrels))


# In[7]:


# mrr(res['preds'][:500], res['ranker_test_all_label_ids'][:500], [50] * 10)


# In[ ]:





# In[11]:


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
    index_rank = np.argsort(-np.asarray(preds))
    for rank, i in enumerate(index_rank):
        if labels[i] == 1:
            score = 1.0 / (rank + 1.0)
            break
            
    return score


# In[12]:


# mrr(res['preds'][:500], res['ranker_test_all_label_ids'][:500], [50] * 10)


# In[13]:


s, _ = mrr(res['preds'], res['ranker_test_all_label_ids'], [50] * len(qrels))
print(s)


# In[ ]:




