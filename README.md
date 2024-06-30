# Dense-Screening-Feedback

This repository contains the code, data, and run files for the SIGIR 2024 paper ***Dense Retrieval with Continuous Explicit Feedback for Systematic Review Screening Prioritisation***.
____

### Dependencies

The environment is based on **Python 3.8**. We use [`Tevatron (v1)`](https://github.com/texttron/tevatron/tree/tevatron-v1) for training dense retrievers, and [`pyserini=0.21.0`](https://pypi.org/project/pyserini/0.21.0/) with `faiss-cpu=1.7.4` for our main retrieval task with feedback. We also use the [`goldilocks-reproduce`](https://github.com/ielab/goldilocks-reproduce) repository for the active learning baselines. Check the installation guides accordingly.



### Data
We use the CLEF-TAR 2017-2019 collections for Subtask 2. The processed data for this paper are available on [Zenodo]().

### Dense retrieval with explicit feedback

Run `tevatron_pipe.py` for dense retriever training, corpus & query encoding, and retrieval (initial ranking).

```
python tevatron_pipe.py --collection_split clef19_intervention \
                        --model_path ./model/clef19_intervention/biolinkbert_128_256_11 \
                        --q_max_len 128 \
                        --p_max_len 256 \
                        --train_n 11 \
                        --train_epoch 60
```

Run`dense_query_tar.py` for dense retrieval with explicit feedback. Rocchio settings are `(1,1,1)`, `(1,0.5,0.5)`, `(1,0.8,0.2)`, `(1,1,0)`.

```
python dense_query_tar.py --collection_split clef17_test \
                          --model ./models/biolinkbert_128_256 \
                          --n_iteration 20 --top_k 25 \
                          --output_path ./trained_results/a1_b5_c5/biolinkbert_128_256_2/clef17_test \
                          --alpha 1.0 \
                          --beta 0.5 \
                          --gamma 0.5
```


### Baselines

##### BM25+RM3

Run `bm25_baseline.py` with the following command:

```
python bm25_baseline.py --collection_split clef19_intervention \
                        --baseline rm3
```

##### CLEF Runs

We select the previous best CLEF runs as baselines. Check [here](./baseline_results/clef/README.md) for more details.

##### TAR with Active Learning

(Logistic Regression) Run `goldilocks_lr.py` as follows:

```
python goldilocks_lr.py --collection_split clef19_dta
```

(BioLinkBERT) Run `goldilocks_screen.py` as follows:

```
python goldilocks_screen.py --collection_split clef19_intervention
```


### Results



### Contact

If you have any questions, feel free to contact `xinyu.mao` [AT] uq.edu.au (replace [AT] with @).
