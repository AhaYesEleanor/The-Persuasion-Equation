
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import json
import string
import pytextrank
import sys


# In[ ]:


def get_docs1(doctype, posts_collection, tl_comments_collection, key_phrase_exists=True, include_undeltad_posts=False):
    '''
    Takes in doctype 'comment' or 'post' and names of mongodb collections for posts and toplevel comments.
    Can take key_phrase_exists=False to obtain only documents that have not been processed yet.
    Can take include_undeltad_posts=True to output undeltad posts if doctype = post.
    Returns 2 lists of dicts, corresponding to deltad and undeltad documents.
    If doctype='post', returns an empty list for undeltad docs unless include_undeltad_posts=True
    '''
    # getting posts where some delta was awarded
    deltad_post_gen = posts_collection.find( {'tl_comment_delta_parents': {"$exists": True}})
    
    if doctype == 'post':
        deltad_docs = [{'id': post[f'{doctype}_id'], 'text': post[f'{doctype}_text'], 'label': 1} for post in deltad_post_gen]
        undeltad_docs =[]
        if include_undeltad_posts:
            undeltad_post_gen = posts_collection.find( {'tl_comment_delta_parents': {"$exists": False}})
            undeltad_docs = [{'id': post[f'{doctype}_id'], 'text': post[f'{doctype}_text'], 'label': 0} for post in undeltad_post_gen]
        
    if doctype == 'comment':
        # get ids of Top Level Comments that resulted in deltas AND list of all TL Comment IDs for posts where some delta was awarded by OP
        post_ids = [(post['tl_comment_delta_parents'], post['comment_ids']) for post in deltad_post_gen]
        (deltad_tl_comment_ids, all_tl_comment_ids) = zip(*post_comment_ids)

        # flatten lists of lists
        deltad_tl_comment_ids = [item for sublist in deltad_tl_comment_ids for item in sublist]
        all_tl_comment_ids = [item for sublist in all_tl_comment_ids for item in sublist]

        # get ids of TL Comments that did not result in deltas from posts where OP did award deltas
        undeltad_tl_comment_ids = list(set(all_tl_comment_ids) - set(deltad_tl_comment_ids))
        
        # retrieve TL comments resulting in deltas by id
        deltad_tl_comment_gen = tl_comments_collection.find( {'$and': [{'comment_id': {"$in": deltad_tl_comment_ids}},{'key_phrases': {"$exists": key_phrase_exists}}]})
        # retrieve TL comments NOT resulting in deltas
        undeltad_tl_comment_gen = tl_comments_collection.find( {'$and': [{'comment_id': {"$in": undeltad_tl_comment_ids}},{'key_phrases': {"$exists": key_phrase_exists}}]})
        
        #get final doc dictionaries
        deltad_docs = [{'id': comment[f'{doctype}_id'], 'text': comment[f'{doctype}_text'], 'label': 1} for comment in deltad_tl_comment_gen]
        undeltad_docs = [{'id': comment[f'{doctype}_id'], 'text': comment[f'{doctype}_text'], 'label': 0} for comment in undeltad_tl_comment_gen]
    '''
    if doctype == 'post+comment':
        # get combined post+comment documents
        for post in deltad_post_gen:
            deltad_tl_comment_ids = post['tl_comment_delta_parents']
            all_tl_comment_ids = post['comment_ids']
            # get ids of TL Comments that did not result in deltas from posts where OP did award deltas
            undeltad_tl_comment_ids = list(set(all_tl_comment_ids) - set(deltad_tl_comment_ids))
        
            # retrieve TL comments resulting in deltas by id
            deltad_tl_comment_gen = tl_comments_collection.find( {'$and': [{'comment_id': {"$in": deltad_tl_comment_ids}},{'key_phrases': {"$exists": key_phrase_exists}}]})
            # retrieve TL comments NOT resulting in deltas
            undeltad_tl_comment_gen = tl_comments_collection.find( {'$and': [{'comment_id': {"$in": undeltad_tl_comment_ids}},{'key_phrases': {"$exists": key_phrase_exists}}]})
        
        #get final doc dictionaries
        deltad_docs = [{'id': comment[f'{doctype}_id'], 'text': comment[f'{doctype}_text'], 'label': 1} for comment in deltad_tl_comment_gen]
        undeltad_docs = [{'id': comment[f'{doctype}_id'], 'text': comment[f'{doctype}_text'], 'label': 0} for comment in undeltad_tl_comment_gen]
    '''    
    return deltad_docs, undeltad_docs


# In[ ]:


def get_docs(posts_collection, tl_comments_collection, key_phrase_exists=True, include_undeltad_posts=False):
    '''
    Takes in names of mongodb collections for posts and toplevel comments.
    Can take key_phrase_exists=False to obtain only documents that have not been processed yet.
    Can take include_undeltad_posts=True to include undeltad posts.
    Returns 2 lists of dicts, corresponding to deltad and undeltad documents.
    '''
    # getting posts where some delta was awarded
    deltad_post_gen = posts_collection.find( {'tl_comment_delta_parents': {"$exists": True}})
    all_deltad_comments=[]
    all_undeltad_comments = []
    for i, post in enumerate(deltad_post_gen):
        if i % 50 == 0:
            print(f'processing post {i}')
        deltad_tl_comment_ids = post['tl_comment_delta_parents']
        all_tl_comment_ids = post['comment_ids']
        # get ids of TL Comments that did not result in deltas from posts where OP did award deltas
        undeltad_tl_comment_ids = list(set(all_tl_comment_ids) - set(deltad_tl_comment_ids))
        
        # retrieve TL comments resulting in deltas by id
        deltad_tl_comment_gen = tl_comments_collection.find( {'$and': [{'comment_id': {"$in": deltad_tl_comment_ids}},{'key_phrases': {"$exists": key_phrase_exists}}]})
        # retrieve TL comments NOT resulting in deltas
        undeltad_tl_comment_gen = tl_comments_collection.find( {'$and': [{'comment_id': {"$in": undeltad_tl_comment_ids}},{'key_phrases': {"$exists": key_phrase_exists}}]})
        
        #get final comment dictionaries
        deltad_comments = [{'id': comment['comment_id'], 'text': comment['comment_text'], 'label': 1, 'post_id': post['post_id'], 'post_text': post['post_text']} for comment in deltad_tl_comment_gen]
        undeltad_comments = [{'id': comment['comment_id'], 'text': comment['comment_text'], 'label': 0, 'post_id': post['post_id'], 'post_text': post['post_text']} for comment in undeltad_tl_comment_gen]
        
        all_deltad_comments.extend(deltad_comments)
        all_undeltad_comments.extend(undeltad_comments)
        
    return all_deltad_comments, all_undeltad_comments


# In[ ]:


def doc_split(deltad_docs, undeltad_docs, test_ratio=0.2, val_set=True, val_ratio=0.2, rand_state=42): 
    '''
    Takes in list of dicts for deltad docs and list of dicts for undeltad docs.
    Returns train, test, and val lists of dicts for docs, stratified by deltad/undeltad.
    Returned val list will be empty list if val_set=False. 
    test_ratio and val_ratio are in relation to the total doc set size so if val_set=True, 
    the train set proportion will be (1-(test_ratio+val_ratio))
    '''
    np.random.seed(seed=rand_state)
    np.random.shuffle(deltad_docs)
    np.random.shuffle(undeltad_docs)
    if val_set:
        total_ratio = test_ratio + val_ratio

        val_split_d = int(val_ratio*len(deltad_docs))
        val_split_u = int(val_ratio*len(undeltad_docs))

        val_docs = deltad_docs[0:val_split_d]
        val_docs.extend(undeltad_docs[0:val_split_u])

        test_split_d = int(total_ratio * len(deltad_docs))
        test_split_u = int(total_ratio * len(undeltad_docs))

        test_docs = deltad_docs[val_split_d:test_split_d]
        test_docs.extend(undeltad_docs[val_split_u:test_split_u])

        train_docs = deltad_docs[test_split_d::]
        train_docs.extend(undeltad_docs[test_split_u::])

    else:
        test_split_d = int(test_ratio*len(deltad_docs))
        test_split_u = int(test_ratio*len(undeltad_docs))

        test_docs = deltad_docs[0:test_split_d]
        test_docs.extend(undeltad_docs[0:test_split_u])

        train_docs= deltad_docs[test_split_d::]
        train_docs.extend(undeltad_docs[test_split_u::])

        val_docs = []

    return train_docs, test_docs, val_docs


# In[ ]:


def get_fields(list_of_doc_dicts):
    '''
    Simple helper function takes in a list of doc dicts,
    returns separate lists of doc ids, doc_texts, doc_labels and the post id that generated the comment.
    '''
    doc_tuples = [(doc['id'],doc['text'],doc['label'], doc['post_id']) for doc in list_of_doc_dicts]
    (doc_ids, doc_texts, doc_labels, doc_posts) = zip(*doc_tuples)
    return doc_ids, doc_texts, doc_labels, doc_post_ids, doc_post_texts


# In[ ]:


def clean_text(texts):
    '''
    Takes in list of text strings to tokenize, returns cleaned texts,
    with all punctuation and digits stripped and all characters converted to lowercase
    '''
    stemmer_inst = stemmer()
    tokenizer_inst = tokenizer()
    cleaned_texts = []
    for text in texts:
        #strip punctuation and digits from whole text
        to_replace = [punc for punc in string.punctuation+string.digits if punc!="'"]
        translate_dict = {key: ' ' for key in to_replace}
        translate_dict["'"] = ''
        replacement_table = str.maketrans(translate_dict)
        stripped_text = text.translate(replacement_table)
        #lower case text
        lowered_text = stripped_text.lower()
        cleaned_texts.append(lowered_text)
    return cleaned_texts


# In[ ]:


def get_ptr_dicts(list_of_doc_dicts):
    '''
    Simple helper function takes in list of doc dictionaries,
    returns list of PyTextRank-ready dictionaries
    '''
    replacement_table = str.maketrans({'\n': ' ', "'": '', '-': '', '/': ''})
    ptr_dicts = [{'id': doc['id'], 'text': doc['text'].translate(replacement_table)} for doc in list_of_doc_dicts]
    return ptr_dicts


# In[ ]:


def insert_key_phrases_into_db(list_of_doc_dicts, doctype, collection):
    '''
    Takes in list of doc dictionaries and a doctype ('comment' or 'post'), 
    processes each doc with PyTextRank, obtains key phrases and 
    inserts key phrases into document in Mongodb as 'key_phrases' field.
    '''
    path_stage0 = 'stage0.json'
    path_stage1 = 'stage1.json'
    path_stage2 = 'stage2.json'
    path_stage3 = 'stage3.json'
    
    total_docs = len(list_of_doc_dicts)
    
    failed_ids=[]
    for i, doc_dict in enumerate(list_of_doc_dicts):
        if i % 50 == 0:
            print(f'processing {i} of {total_docs} documents')
        doc_dict['text'] = doc_dict['text'].split('\n_____\n\n')[0]

        try:
            with open(path_stage0, 'w') as f:
                json.dump(doc_dict, f)
            # Stage 1    
            with open(path_stage1, 'w') as f:
                for graf in pytextrank.parse_doc(pytextrank.json_iter(path_stage0)):
                    f.write("%s\n" % pytextrank.pretty_print(graf._asdict()))
                    # print(pytextrank.pretty_print(graf))
            # Stage 2
            graph, ranks = pytextrank.text_rank(path_stage1)
            pytextrank.render_ranks(graph, ranks)
            with open(path_stage2, 'w') as f:
                for rl in pytextrank.normalize_key_phrases(path_stage1, ranks):
                    f.write("%s\n" % pytextrank.pretty_print(rl._asdict()))
                    # to view output in this notebook
                    # print(pytextrank.pretty_print(rl))
            # Stage 3
            kernel = pytextrank.rank_kernel(path_stage2)
            with open(path_stage3, 'w') as f:
                for s in pytextrank.top_sentences(kernel, path_stage1):
                    f.write(pytextrank.pretty_print(s._asdict()))
                    f.write("\n")
                    # to view output in this notebook
                    # print(pytextrank.pretty_print(s._asdict()))
            # Stage 4
            phrase_list = list(set([p for p in pytextrank.limit_keyphrases(path_stage2, phrase_limit=15)]))
            phrases = ", ".join(phrase_list)

            sent_iter = sorted(pytextrank.limit_sentences(path_stage3, word_limit=150), key=lambda x: x[1])
            s = []

            for sent_text, idx in sent_iter:
                s.append(pytextrank.make_sentence(sent_text))

            graf_text = " ".join(s)
            collection.update_one({f'{doctype}_id': {'$eq': doc_dict['id']}},{'$set': {'key_phrases': phrase_list}})
        except:
            failed_ids.append(doc_dict['id'])
            print('failed on ',doc_dict['id'])
            continue


# In[ ]:


def get_all_key_phrases(collection, doctype, doc_id_list):
    '''
    Takes in a mongodb collection, doctype ('post' or 'collection'),
    finds all docs in said collection that have a key_phrases field and whose doc_id is in doc_id_list.
    Returns array of the set of all key phrases present in these docs-to be passed to CountVectorizer as a vocabulary.
    '''
    key_phrase_gen = collection.find({'$and': [{f'{doctype}_id': {"$in": doc_id_list}},{'key_phrases': {"$exists": True}}]})
    key_phrases_list = [comment['key_phrases'] for comment in key_phrase_gen]
    num_docs = len(key_phrases_list)
    flat_key_phrases = [item for sublist in key_phrases_list for item in sublist]
    key_phrases_list = list(set(flat_key_phrases))
    print(f'{len(key_phrases_list)} unique key phrases from {num_docs} documents')
    phrase_array = np.array(key_phrases_list)
    
    return phrase_array

