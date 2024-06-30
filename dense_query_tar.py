import os 
import pickle
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from hashlib import md5
import argparse

from faiss import read_index
from pyserini.search import FaissSearcher, DenseVectorAveragePrf
from rf_rocchio import RocchioRf
from utils import *


def perform_relevance_feedback(searcher, query_embedding, topic, n_docs, args, model_name=None):
    output_path = args.output_path
    fix_path(output_path)
    reviewed_docs = set()
    reviewed_ranks = []
    tag = f'{topic}'
    total_record = list()
    top_k = args.top_k
    if args.auto_batch_size:
        top_k = 1
    
    # Load labels for relevancy judgment.
    if '19' in args.collection_split:
        collection_year, review_type, split_name = args.collection_split.split('_')
        rel_table = pd.read_pickle(f'./clef_info/{collection_year}/{review_type}/{split_name}/{topic}/rel_info.pkl')
    else:
        collection_year, split_name = args.collection_split.split('_')
        rel_table = pd.read_pickle(f'./clef_info/{collection_year}/{split_name}/{topic}/rel_info.pkl')
    num_total_pos = rel_table.iloc[:, 1].sum()
    
    if args.verbose:
        print(f"Topic {topic}, Total docs {len(rel_table)}, Total relevant docs {num_total_pos}")

    # Record time for each topic
    # start_time = time.time()
    if args.n_iteration == 0:
        query_embedding, search_result = searcher.search(query_embedding, n_docs, return_vector=True)
        record = result_record(search_result, query_embedding)
        pickle.dump(record, (output_path / f"{tag}_it_0_total.results.pkl.saving").open('wb'))
        (output_path / f"{tag}_it_0_total.results.pkl.saving").rename((output_path / f"{tag}_it_0_total.results.pkl"))
    else:
        if args.n_seed > 0:
            # Randomly select seed documents.
            hashs = pd.Series(rel_table.index.astype(str).map(lambda x: md5(x.encode()).hexdigest()),
                              index=rel_table.index, name='md5')
            rel_info = rel_table.assign(md5=hashs).reset_index(drop=True)
            sorted_rel = rel_info.sort_values('md5')
            seedset = sorted_rel.pmid[sorted_rel.iloc[:, 1]].tolist()[:args.n_seed]
            reviewed_docs.update(seedset)

            # Get seed document embeddings.
            index = read_index(f"indexes/{model_name}/{args.collection_split}/{topic}/index")
            seed_embeddings = []
            for doc_id in seedset:
                doc_idx_list = searcher.docids
                doc_idx = doc_idx_list.index(doc_id)
                seed_embeddings.append(index.reconstruct(doc_idx))
            seed_embeddings = np.array(seed_embeddings)

            # Update query embedding with seed embeddings.
            seed_q_embedding = get_seed_q_embs(seed_embeddings, query_embedding)
            query_embedding, search_result = searcher.search(seed_q_embedding, n_docs, return_vector=True)
            search_result = remove_reviewed_docs(search_result, reviewed_docs)
            record = result_record(search_result=search_result, query_embedding=query_embedding, use_seed_q=seed_embeddings)
            print(f'Finished Iteration 0, with seed doc {seedset}')

        else:
            query_embedding, search_result = searcher.search(query_embedding, n_docs, return_vector=True)
            if args.prf_init:
                top_docs = search_result[:top_k]
                avg_prf = DenseVectorAveragePrf()
                prf_init_query_embedding = avg_prf.get_prf_q_emb(query_embedding, prf_candidates=top_docs)
                prf_init_query_embedding = prf_init_query_embedding.reshape((1, len(prf_init_query_embedding)))
                print(f"Initialised with {top_k} prf doc embeddings.")
                query_embedding, search_result = searcher.search(prf_init_query_embedding, n_docs, return_vector=True)

            record = result_record(search_result, query_embedding, exhaust=args.exhaust_docs)
            print('Finished Iteration 0...')
        
        if args.store_first_iteration:
            pickle.dump(record, (output_path / f"{tag}_it_0.results.pkl.saving").open('wb'))
            (output_path / f"{tag}_it_0.results.pkl.saving").rename((output_path / f"{tag}_it_0.results.pkl"))

        if args.exhaust_docs:
            args.n_iteration = math.ceil(n_docs / top_k)
            print(f"To exhaust Topic {topic} with Total docs {n_docs} in {args.n_iteration} iterations.")

        for i in range(args.n_iteration):
            # Select top k documents to review for feedback.
            top_k += int(np.ceil(top_k/10))
            print(f"Iteration {i}, current batch size {top_k}")
            top_docs = search_result[:top_k]
            top_doc_ids = get_doc_ids(top_docs)
            reviewed_docs.update(top_doc_ids)
            reviewed_ranks.append(top_docs)
            num_left_docs = n_docs - len(reviewed_docs)

            # Perform relevance feedback.
            # Handel the case when there are not enough documents to perform RF.
            if num_left_docs != 0:
                if num_left_docs < top_k:
                    print(f"After reviewing, only {num_left_docs} docs left for next iteration.")
                if args.method == "avg":
                    # update query vector with top k pseudo relevance feedback
                    avg_prf = DenseVectorAveragePrf()
                    new_query_embedding = avg_prf.get_prf_q_emb(query_embedding, prf_candidates=top_docs)
                elif args.method == "rocchio":
                    # update query vector with top k relevance feedback
                    rocchio_rf = RocchioRf(alpha=args.alpha, beta=args.beta, gamma=args.gamma, top_k=top_k, rel_table=rel_table)
                    new_query_embedding = rocchio_rf.get_rf_q_emb(query_embedding, rf_candidates=top_docs)
                else:
                    raise NotImplementedError()
                if len(new_query_embedding.shape) == 1:
                    query_embedding = new_query_embedding.reshape((1, len(new_query_embedding)))
                else:
                    query_embedding = new_query_embedding
            else:
                print("Docs exhausted.")
                break
                
            del search_result

            # Rerank the documents with updated query
            _, search_result = searcher.search(query_embedding, n_docs, return_vector=True)

            # Remove all reviewed documents
            search_result = remove_reviewed_docs(search_result, reviewed_docs)
            num_left_docs = n_docs - len(reviewed_docs)

            record = result_record(search_result, query_embedding=query_embedding, record_doc=args.record_doc, exhaust=args.exhaust_docs)

            if args.verbose:
                print(f'Finished Iteration {i + 1}...')
                print(f'Total {n_docs}, {num_left_docs} documents not reviewed')

            # Save result in each iteration
            if args.save_iteration_result:
                pickle.dump(record, (output_path / f"{tag}_it_{i+1}.results.pkl.saving").open('wb'))
                (output_path / f"{tag}_it_{i+1}.results.pkl.saving").rename((output_path / f"{tag}_it_{i+1}.results.pkl"))
            else:
                total_record.append(record)

        # end_time = time.time()
        # topic_time = end_time - start_time
        # print(f'Completed {topic} in {topic_time/60} min')
        if args.exhaust_docs:
            pickle.dump((total_record, reviewed_ranks), (output_path / f"{tag}_total.results.pkl.saving").open('wb'))
            (output_path / f"{tag}_total.results.pkl.saving").rename((output_path / f"{tag}_total.results.pkl"))
        else:
            pickle.dump((total_record, reviewed_docs), (output_path / f"{tag}_total.results.pkl.saving").open('wb'))
            (output_path / f"{tag}_total.results.pkl.saving").rename((output_path / f"{tag}_total.results.pkl"))



def main(args):
    # Initialisation: seed(all relevant) or dense retrieval(dense query)
    # Load query
    if '/' in args.model_path:
        model_name = args.model_path.split('/')[-1].lower()
    else:
        model_name = args.model_path
    query_embeddings = pd.read_pickle(f"queries/encoding/{model_name}/{args.collection_split}.queries.pkl")

    # Load indexed docs for each topic within the collection
    topic_list = list(query_embeddings.keys())

    # For seed setting, exclude topic with only one relevant doc
    if args.n_seed and args.collection_split == "clef19_intervention_train":
        topic_list.remove('CD010019')

    start_time = time.time()
    for topic in tqdm(topic_list):
        # If seed, randomly n=1 relevant
        # If dense, relevance feedback with rocchio settings
        searcher = FaissSearcher(
                   f"indexes/{model_name}/{args.collection_split}/{topic}",
                   args.model_path)
        q_emb = query_embeddings[topic]
        if len(q_emb.shape) == 1:
            q_emb = q_emb.reshape((1, len(q_emb)))
        n_docs = searcher.num_docs
        # Run tar and record result
        perform_relevance_feedback(searcher=searcher, query_embedding=q_emb, topic=topic, n_docs=n_docs, args=args, model_name=model_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Completed {args.collection_split} in {elapsed_time/60} min')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dense query TAR with feedback setting
    parser.add_argument("--collection_split", default="clef17_test", type=str, help="clef17_test, clef17_train, "
                                                                                    "clef18_test, clef19_dta_test, "
                                                                                    "clef19_intervention_test, "
                                                                                    "clef19_intervention_train")
    parser.add_argument("--model_path", default="bert-base-uncased", type=str, help="huggingface encoders")
    parser.add_argument('--method', type=str, default='rocchio')
    parser.add_argument('--n_seed', type=int, default=0)
    parser.add_argument("--prf_init", action="store_true", default=False)
    parser.add_argument('--n_iteration', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=3)

    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--disable_tqdm', action='store_true', default=False)

    parser.add_argument("--store_first_iteration", action="store_true", default=True)
    parser.add_argument('--output_path', type=Path, default='./results/bert-base-uncased/clef17_test')
    parser.add_argument("--save_iteration_result", action="store_true", default=False)
    parser.add_argument("--record_doc", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--exhaust_docs", action="store_true", default=False)


    args = parser.parse_args()

    main(args)
