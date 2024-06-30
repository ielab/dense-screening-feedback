import numpy as np
from typing import List, Dict
from pyserini.search.faiss import PRFDenseSearchResult


class DenseVectorPrf:
    def __init__(self):
        pass

    def get_prf_q_emb(self, **kwargs):
        pass

    def get_batch_prf_q_emb(self, **kwargs):
        pass


class RocchioRf(DenseVectorPrf):
    def __init__(self, alpha: float, beta: float, gamma: float, top_k: int, rel_table):
        """
        Parameters
        ----------
        alpha : float
            Rocchio parameter, controls the weight assigned to the original query embedding.
        beta : float
            Rocchio parameter, controls the weight assigned to the positive document embeddings.
        gamma : float
            Rocchio parameter, controls the weight assigned to the negative document embeddings.
        top_k : int
            Rocchio parameter, select top k documents to review and get relevant documents as positive feedbacks while irrelevant documents as negative feedbacks.
        """
        DenseVectorPrf.__init__(self)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._top_k = top_k
        self.rel_table = rel_table

    def get_top_k(self):
        return self._top_k

    def set_top_k(self, value):
        if value >= 0:
            self._top_k = value
        else:
            raise ValueError("top_k must be a non-negative value")

    def get_rf_q_emb(self, emb_qs: np.ndarray = None, rf_candidates: List[PRFDenseSearchResult] = None):
        """Perform Rocchio RF with Dense Vectors

        Parameters
        ----------
        emb_qs : np.ndarray
            query embedding
        rf_candidates : List[PRFDenseSearchResult]
            List of PRFDenseSearchResult, contains document embeddings.

        Returns
        -------
        np.ndarray
            return new query embeddings
        """
        pos_doc_embs, neg_doc_embs = self.get_rf_d_embs(rf_candidates, self.rel_table)
        weighted_query_embs = self.alpha * emb_qs
        weighted_mean_pos_doc_embs = self.beta * np.mean(pos_doc_embs, axis=0)
        new_emb_q = weighted_query_embs + weighted_mean_pos_doc_embs
        weighted_mean_neg_doc_embs = self.gamma * np.mean(neg_doc_embs, axis=0)
        new_emb_q -= weighted_mean_neg_doc_embs
        return new_emb_q

    def get_batch_rf_q_emb(self, topic_ids: List[str] = None, emb_qs: np.ndarray = None,
                            rf_candidates: Dict[str, List[PRFDenseSearchResult]] = None):
        """Perform Rocchio RF with Dense Vectors in Batch

        Parameters
        ----------
        topic_ids : List[str]
            List of topic ids.
        emb_qs : np.ndarray
            Query embeddings
        rf_candidates : List[PRFDenseSearchResult]
            List of PRFDenseSearchResult, contains document embeddings.

        Returns
        -------
        np.ndarray
            return new query embeddings
        """
        qids = list()
        new_emb_qs = list()
        for index, topic_id in enumerate(topic_ids):
            qids.append(topic_id)
            new_emb_qs.append(self.get_rf_q_emb(emb_qs[index], rf_candidates[topic_id]))
        new_emb_qs = np.array(new_emb_qs).astype('float32')
        return new_emb_qs
    
    def get_rf_d_embs(self, rf_candidates, rel_table):
        pos_doc_embs = []
        neg_doc_embs = []
        emb_dim = rf_candidates[0].vectors.shape
        
        for doc in rf_candidates:
            if rel_table[rel_table['pmid'] == doc.docid].iloc[:, 1].values[0]:
                pos_doc_embs.append(doc.vectors) 
            elif not rel_table[rel_table['pmid'] == doc.docid].iloc[:, 1].values[0]:
                neg_doc_embs.append(doc.vectors)
        if len(pos_doc_embs)+len(neg_doc_embs) != self.get_top_k():
            print(f"current pos docs: {len(pos_doc_embs)}, neg docs: {len(neg_doc_embs)}, but relevance feedback: {self.get_top_k()}")
            assert len(pos_doc_embs)+len(neg_doc_embs) == self.get_top_k()

        if len(pos_doc_embs) == 0:
            pos_doc_embs = np.zeros(emb_dim)
        elif len(neg_doc_embs) == 0:
            neg_doc_embs = np.zeros(emb_dim)
        return np.array(pos_doc_embs), np.array(neg_doc_embs)