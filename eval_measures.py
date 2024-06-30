import pandas as pd
import numpy as np
from analysis_tools import get_rel_ranks, get_rel_table


def get_wss_at_target_recall(pos_rank, pos_total, num_left_docs, target_recall=1.0):
    '''
    pos_rank: the rank of relevant docs
    target recall: 1, 0.95, 0.9
    '''
    current_recall = get_recall(pos_rank, pos_total, target_recall=target_recall)
    # if target recall already reached
    if target_recall-current_recall < 0:
        wss_at_target_recall = 1
    else:
        num_pos_target = int(np.ceil(pos_total*target_recall) - len(pos_rank))
        pos_target_rank = pos_rank[num_pos_target-1]
        dfr = pos_target_rank/num_left_docs
        wss_at_target_recall = target_recall - (dfr/num_left_docs)

    return wss_at_target_recall


def get_recall(reviewed_doc_ids, num_total_pos, rel_table):
    
    # find the corresponding index in rel table for ranked docs
    table_idx = [rel_table[rel_table['pmid']==hit_id].index.values[0] for hit_id in reviewed_doc_ids]

    # find the rank of relevant docs
    pos_reviewed_ids = []
    for doc_id in reviewed_doc_ids:
        if rel_table[rel_table['pmid'] == doc_id].iloc[:,1].values[0] == True:
            pos_reviewed_ids.append(doc_id)
    recall = len(pos_reviewed_ids)/num_total_pos
    return recall


def get_precision(top_doc_ids, rel_table):
    '''
    top_k_docs: the top k of docs for relevance feedback
    '''
    # get the number of relevant docs in top k
    pos_in_review = len(get_rel_ranks(top_doc_ids, rel_table))

    # get the precision@k
    precision = pos_in_review/len(top_doc_ids)

    return precision

def get_last_rel_init(model_name, collection_split):
    # get last rank on init
    init_dirs = [dir_ for dir_ in glob.glob(f'./results/{model_name}/{collection_split}/*it_0.results.pkl')]
    last_dict_init = {}
    for dir in init_dirs:
        topic = dir.split('/')[-1][:8]
        rel_table = get_rel_table(collection_split, topic)
        result_init = pd.read_pickle(dir)
        last_rel_init = get_rel_ranks(result_init[0]['doc_id'], rel_table)[-1]
        last_dict_init[topic] = [last_rel_init]
        
    return last_dict_init

def get_last_rel(model_name, collection_split):
    # get last rank on init
    last_dict = get_last_rel_init(model_name, collection_split)

    # get last rank on the runs
    for topic in tqdm(list(last_dict.keys())):
        rel_table = get_rel_table(collection_split, topic)
        total_docs = len(rel_table)
        result_total = pd.read_pickle(f'./results/{model_name}/{collection_split}/{topic}_total.results.pkl')
        for i in range(len(result_total[0])):
            rel_rank = get_rel_ranks(result_total[0][i][0]['doc_id'], rel_table)
            num_unreviewed = len(result_total[0][i][0]['doc_id'])
            num_reviewed = total_docs - num_unreviewed
            if len(rel_rank) != 0:
                last_rel = rel_rank[-1]
                total_last = num_reviewed + last_rel
                last_dict[topic].append(total_last)
            else:
                pass
    return last_dict