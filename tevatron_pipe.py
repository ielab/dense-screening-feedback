from tqdm import tqdm
import pandas as pd
import argparse
import subprocess
from timeit import default_timer as timer
import os
from utils import generate_trec_run, generate_tar_eval


def dpr_pipeline(collection_split, model_path, q_max_len, p_max_len, train_n, train_epoch):
    model_names = ['biolinkbert', 'pubmedbert', 'biobert', 'co-condenser']
    model_name = ''
    for name in model_names:
        if name in model_path.lower():
            model_name = name
        else:
            continue
    if model_name == '':
           raise ValueError(f'Unknown model from {model_path}')

    train_model_dir = f'model/{collection_split}/{model_name}_{q_max_len}_{p_max_len}_{train_n}'
    
    # Tevatron train
    cmd_train = f'python -m tevatron.driver.train \
       --output_dir {train_model_dir} \
       --dataset_name Tevatron/msmarco-passage \
       --train_dir dpr_train/{collection_split} \
       --model_name_or_path {model_path} \
       --do_train \
       --save_steps 200 \
       --fp16 \
       --train_n_passages {train_n} \
       --learning_rate 1e-5 \
       --q_max_len {q_max_len} \
       --p_max_len {p_max_len} \
       --num_train_epochs {train_epoch} \
       --overwrite_output_dir \
    '
    subprocess.run(cmd_train, shell=True, check=True)

    # Tevatron corpus
    corpus_encode_path = f'tevatron_corpus_encode/{collection_split}/{model_name}_{q_max_len}_{p_max_len}_{train_n}'
    if not os.path.exists(corpus_encode_path):
        os.makedirs(corpus_encode_path)

    cmd_encode_corpus = f'python -m tevatron.driver.encode \
       --model_name_or_path {train_model_dir} \
       --fp16 \
       --per_device_eval_batch_size 256 \
       --p_max_len 128 \
       --dataset_name Tevatron/msmarco-passage-corpus \
       --encode_in_path corpus_dir/{collection_split}_test.jsonl  \
       --encoded_save_path {corpus_encode_path}/{collection_split}_corpus.pkl \
       --output_dir=temp \
    '
    subprocess.run(cmd_encode_corpus, shell=True, check=True)

    n_docs = len(pd.read_pickle(f'{corpus_encode_path}/{collection_split}_corpus.pkl')[1])

    query_encode_path = f'tevatron_queries_encode/{collection_split}/{model_name}_{q_max_len}_{p_max_len}_{train_n}/'
    if not os.path.exists(query_encode_path):
        os.makedirs(query_encode_path)

    # Tevatron query
    cmd_encode_q = f'python -m tevatron.driver.encode \
       --model_name_or_path  model/{collection_split}/{model_name}_{q_max_len}_{p_max_len}_{train_n} \
       --fp16 \
       --p_max_len 128 \
       --dataset_name Tevatron/msmarco-passage/dev \
       --encode_in_path dev_dir/{collection_split}_test_query_obj.jsonl  \
       --encode_is_qry \
       --encoded_save_path {query_encode_path}/{collection_split}_query_obj.pkl \
       --output_dir=temp \
    '
    subprocess.run(cmd_encode_q, shell=True, check=True)

    result_path = f'./tevatron_results/{collection_split}/{model_name}_{q_max_len}_{p_max_len}_{train_n}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # Tevatron retrieval
    cmd_retreive = f'python -m tevatron.faiss_retriever \
       --query_reps {query_encode_path}/{collection_split}_query_obj.pkl \
       --passage_reps {corpus_encode_path}/{collection_split}_corpus.pkl \
       --depth {n_docs} \
       --batch_size -1 \
       --save_text \
       --save_ranking_to {result_path}/{collection_split}_rank_obj.txt \
    '
    subprocess.run(cmd_retreive, shell=True, check=True)
    
    generate_trec_run(collection_split, result_path)
    
    generate_tar_eval(collection_split, result_path, model_name, q_max_len, p_max_len, train_n)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_split", default="clef17", type=str, help="clef17, clef18, clef19_dta, clef19_intervention")
    parser.add_argument("--model_path", default="michiyasunaga/BioLinkBERT-base", type=str, help="biolinkbert, pubmedbert, biobert")
    parser.add_argument("--q_max_len", default=128, type=int)
    parser.add_argument("--p_max_len", default=256, type=int)
    parser.add_argument("--train_n", default=11, type=int, help="2,6,11")
    parser.add_argument("--train_epoch", default=10, type=int)
    
    args = parser.parse_args()
    
    start_timer = timer()
    dpr_pipeline(args.collection_split, args.model_path, args.q_max_len, args.p_max_len, args.train_n, args.train_epoch)
    end_timer = timer()
    print(f'Time used: {end_timer-start_timer} seconds.')