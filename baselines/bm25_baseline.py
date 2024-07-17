import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from pyserini.search.lucene import LuceneSearcher
from utils import *


def main(args):
    # load topics and queries
    if '19' in args.collection_split:
        year, type_ = args.collection_split.split('_')
        query_file = pd.read_csv(f"./queries/{year}.{type_}.test.beir.tsv", sep='\t', header=None)
        query_dict = dict(zip(query_file[0], query_file[1]))
        topic_list = query_file[0].tolist()
    else:
        query_file = pd.read_csv(f"./queries/{args.collection_split}.test.beir.tsv", sep='\t', header=None)
        query_dict = dict(zip(query_file[0], query_file[1]))
        topic_list = query_file[0].tolist()

    start_time = time.time()
    for topic in tqdm(topic_list):
        query = query_dict[topic]
        searcher = LuceneSearcher(
            f"bm25_indexes/{args.collection_split}_test_collection/{topic}")
        n_docs = searcher.num_docs

        if args.baseline == "bm25":
            searcher.set_bm25(k1=0.9, b=0.4)
        elif args.baseline == "rm3":
            searcher.set_rm3(10, 10, 0.5, debug=True)
        elif args.baseline == "rocchio":
            searcher.set_rocchio(debug=True)
        else:
            raise ValueError(f"Unknown baseline type {args.baseline}")

        hits = searcher.search(q=query, k=n_docs)

        with open(f'baseline_results/{args.baseline}/{args.collection_split}.run', "a") as f:
            for i, hit in enumerate(hits):
                f.write(f"{topic} 0 {hit.docid} {i} {hit.score} {args.baseline}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Completed {args.collection_split} in {elapsed_time / 60} min')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--collection_split", default="clef17", type=str)
    parser.add_argument('--n_iteration', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--disable_tqdm', action='store_true', default=False)
    parser.add_argument("--store_first_iteration", action="store_true", default=True)
    parser.add_argument("--save_iteration_result", action="store_true", default=False)
    parser.add_argument("--record_doc", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--baseline", default="bm25", type=str, help="bm25, rm3, rocchio")

    args = parser.parse_args()
    main(args)
