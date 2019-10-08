#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
from tqdm import tqdm
from time import sleep


# In[2]:


test_in = '/mnt/scratch/chenqu/aol/original/session_test.txt'
test_out = '/mnt/scratch/chenqu/aol/original/session_test.gen.encoding.txt'


# In[3]:


url = 'https://api.msturing.org/gen/encode'
params = {'key': 'value'}
headers = {'Content-Type': 'application/json',
           'Ocp-Apim-Subscription-Key': '5d0ffce20b164037945a43206857e84a'}
# queries_for_request = {"queries": ["Microsoft", "Bing.com", "Azure Cloud Services"]}
# r = requests.post(url, headers=headers, json=queries_for_request)


# In[4]:


# send a request every N sessions
N = 10
counter = 0
all_queries = []
query_num = []
session_ids = []
error_all_queries = []
error_session_ids = []
error_query_num = []
with open(test_in) as fin, open(test_out, 'w') as fout:
    for line in tqdm(fin):
        counter += 1
        session = json.loads(line.strip())
        session_id = session['session_id']
        session_ids.append(session_id)
        queries = [turn['text'] for turn in session['query']]
        query_num.append(len(queries))
        all_queries.extend(queries)
        
        if counter == N:
            queries_for_request = {'queries': all_queries}
            r = requests.post(url, headers=headers, json=queries_for_request)
            query_vectors = []
            if r.status_code == 200:
                res = r.json()
                for item in res:
                    if item['status'] == 'SUCCESS':
                        query_vectors.append(item['vector'])
                    else:
                        print('failed query')
                        print('session_ids', session_ids)
                        print('response', item)
                
                for num, session_id in zip(query_num, session_ids):
                    each_query_vectors = query_vectors[:num]
                    query_vectors = query_vectors[num:]
                    fout.write(json.dumps({'vectors': each_query_vectors, 'session_id': session_id}) + '\n') 

            else:
                print('wrong status code', r.status_code)
                print('response content', r.content)
                print('session_ids', session_ids)
                print('queries', queries)
                error_all_queries.extend(all_queries)
                error_session_ids.extend(session_ids)
                error_query_num.extend(query_num)
                
                
            all_queries, query_num, session_ids = [], [], []
            counter = 0
            
        sleep(0.1)


# In[5]:


print('dealing with error queries')
with open(test_out, 'a') as fout:
    for num, session_id in tqdm(zip(error_query_num, error_session_ids)):
        error_queries = error_all_queries[:num]
        error_all_queries = error_all_queries[num:]
        queries_for_request = {'queries': error_queries}
        r = requests.post(url, headers=headers, json=queries_for_request)
        query_vectors = []
        if r.status_code == 200:
            res = r.json()
            for item in res:
                if item['status'] == 'SUCCESS':
                    query_vectors.append(item['vector'])
                else:
                    print('failed query')
                    print('session_id', session_id)
                    print('response', item)

            fout.write(json.dumps({'vectors': query_vectors, 'session_id': session_id}) + '\n')
            
        else:
            print('wrong status code', r.status_code)
            print('response content', r.content)
            print('session_id', session_id)
            print('error_queries', error_queries)
            
        sleep(0.1)

