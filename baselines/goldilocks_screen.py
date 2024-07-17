import glob
from tqdm import tqdm
from timeit import default_timer as timer
import pickle
from pathlib import Path
import argparse
import subprocess


def get_topics(clef_year, clef_type):
    if '19' in clef_year:
        dirs = [dir_ for dir_ in glob.glob(f'./clef_info/{clef_year}/{clef_type}/test/*')]
    else:
        dirs = [dir_ for dir_ in glob.glob(f'./clef_info/{clef_year}/test/*')]
    topics = []
    for dir in dirs:
        topic = dir.split('/')[-1][:8]
        topics.append(topic)
    
    return topics


def goldilocks_screening(collection_split):

    if '19' in collection_split:
        collection_name, clef_type = collection_split.split('_')
        clef_year = '20'+collection_name[-2:]
    
    else:
        clef_type = None
        clef_year = '20'+collection_split[-2:]
    
    topic_list = get_topics(clef_year, clef_type)
    topic_time = {}
    
    for topic in tqdm(topic_list):
        
        start_timer = timer()
        
        if '19' in collection_split:
            # 2019 dta, intervention test
            cmd_al = f'python3 {collection_split}_biolink_exp.py --topic {topic} \
            --cached_dataset ./cache_new/clef_biolink/clef{clef_year}_{clef_type}_test_{topic}_biolink_bert-base.512.pkl.gz \
            --dataset_path  ./clef_info/{clef_year}/{clef_type}/test/{topic} \
            --output_path  ./results/biolink/20ep/{collection_name}_{clef_type}_test/ \
            --batch_size 25 \
            --model_path  michiyasunaga/BioLinkBERT-base \
            --al_epoch 20 \
            --sampling_strategy relevance \
            '
            subprocess.run(cmd_al, shell=True, check=True)
        
        else:
            # 2017, 2018 test
            cmd_al = f'python3 {collection_split}_biolink_exp.py --topic {topic} \
            --cached_dataset ./cache_new/clef_biolink/clef{clef_year}_test_{topic}_biolink_bert-base.512.pkl.gz \
            --dataset_path  ./clef_info/{clef_year}/test/{topic} \
            --output_path  ./results/biolink/20ep/{collection_split}_test/ \
            --batch_size 25 \
            --model_path  michiyasunaga/BioLinkBERT-base \
            --al_epoch 20 \
            --sampling_strategy relevance \
            '
            subprocess.run(cmd_al, shell=True, check=True)
            
        end_timer = timer()
        topic_time[topic] = float(end_timer-start_timer)
        print(f'Topic:{topic}, Time used: {end_timer-start_timer} seconds.')


    if '19' in collection_split:
        output_path = Path(f"./results/biolink/20ep/{collection_name}_{clef_type}_test")
        pickle.dump((topic_time), (output_path / f"{collection_split}.time.pkl").open('wb'))
    else:
        output_path = Path(f"./results/biolink/20ep/{collection_split}_test")
        pickle.dump((topic_time), (output_path / f"{collection_split}.time.pkl").open('wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_split", default="clef17", type=str, help="clef17, clef18, clef19_dta, clef19_intervention")
    
    args = parser.parse_args()
    goldilocks_screening(args.collection_split)
