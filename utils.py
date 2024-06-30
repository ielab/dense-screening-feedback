# For tevatron pipe
def generate_trec_run(collection_split, result_path):
    topic_doc_dict = {}

    if '19' not in collection_split:
        queries = pd.read_table(f"./queries/{collection_split}.test.obj.beir.tsv", sep='\t', header=None,
                                names=['topic', 'title'])
        collection_dir = f'{collection_split}'
    else:
        collection_name = collection_split.split('_')
        queries = pd.read_table(f"./queries/{collection_name[0]}.{collection_name[1]}.test.obj.beir.tsv", sep='\t',
                                header=None, names=['topic', 'title'])
        collection_dir = f'{collection_name[0]}/{collection_name[1]}'

    query_list = set(queries.topic.to_list())

    for topic in tqdm(query_list):
        print(topic)
        rel_table = pd.read_pickle(f'./clef_info/{collection_dir}/test/{topic}/rel_info.pkl')
        topic_doc_ids = set(rel_table['pmid'].to_list())
        topic_doc_dict[topic] = topic_doc_ids

    run_file = f'{result_path}/dpr_{collection_split}_rank_obj'
    orgn_file = f'{result_path}/{collection_split}_rank_obj.txt'

    with open(run_file, "w") as g:
        with open(orgn_file, "r") as f:
            i = 0
            for line in tqdm(f):
                topic, doc_id, score = line.split('	')
                doc_ids = topic_doc_dict[topic]
                if doc_id in doc_ids:
                    g.write(f"{topic} 0 {doc_id} {i} {float(score)} dpr_{collection_split}_test\n")
                    i += 1
                else:
                    continue
    print('Run file generated')
    return


def generate_tar_eval(collection_split, result_path, model_name, q_max_len, p_max_len, train_n):
    if '19' in collection_split:
        collection_name = collection_split.replace('_', '.')
        qrel_path = f'./tar_eval/clef_qrels/{collection_name}.test.qrels'
    else:
        qrel_path = f'./tar_eval/clef_qrels/{collection_split}.test.qrels'

    if '17' in collection_split:
        cmd_eval = f'python3 ./tar_eval/tar_eval.py \
    {qrel_path} \
    {result_path}/dpr_{collection_split}_rank_obj >  {result_path}/{model_name}_{q_max_len}_{p_max_len}_{train_n}_obj.res'
    else:
        cmd_eval = f'python3 ./tar_eval/tar_eval_2018.py 2 \
    {qrel_path} \
    {result_path}/dpr_{collection_split}_rank_obj >  {result_path}/{model_name}_{q_max_len}_{p_max_len}_{train_n}_obj.res'

    subprocess.run(cmd_eval, shell=True, check=True)

    with open(f'{result_path}/{model_name}_{q_max_len}_{p_max_len}_{train_n}_obj.res', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'ALL	ap' in line:
                ap = line.strip('\n').split('\t')[-1]
            elif 'ALL	last_rel' in line:
                last_rel = line.strip('\n').split('\t')[-1]
    print(ap, last_rel)


# For dense tar, baseline
def result_record(search_result, query_embedding, record_doc=True, use_seed_q=None, exhaust=False):
    scores = []
    doc_ids = []
    q_embeddings = query_embedding
    doc_embeddings = []
    record = []

    for i in range(len(search_result)):
        doc_ids.append(search_result[i].docid)
        scores.append(search_result[i].score)
        if record_doc:
            doc_embeddings.append(search_result[i].vectors)

    if exhaust:
        record.append({
            'doc_id': doc_ids,
            'score': scores,
        })
    else:
        record.append({
            'doc_id': doc_ids,
            'score': scores,
            'query_embedding': q_embeddings,
            'doc_embedding': doc_embeddings,
        })

    if use_seed_q is not None:
        record.append({
            'seed_q_embedding': use_seed_q,
        })
    return record


def get_doc_ids(search_result):
    doc_id_list = [doc.docid for doc in search_result]
    return doc_id_list


def get_seed_q_embs(seed_embs, query_emb):
    emb_dim = query_emb.shape
    new_query_emb = query_emb

    mean_seed_embs = np.mean(seed_embs, axis=0)
    new_query_emb += mean_seed_embs

    # if len(q_emb.shape) == 1:
    #     q_emb = q_emb.reshape((1, len(q_emb)))
    assert new_query_emb.shape == emb_dim
    return new_query_emb


def remove_reviewed_docs(search_result, reviewed_docs):
    # create new one instead of modifying to avoid unexpected behaviour.
    after_review_docs = []
    for doc in search_result:
        if doc.docid not in reviewed_docs:
            after_review_docs.append(doc)
    return after_review_docs


def fix_path(path):
    if not os.path.exists(path):
        os.makedirs(path)