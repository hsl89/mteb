import logging
from time import time
from typing import Dict, List

import torch.multiprocessing as mp

from sentence_transformers import SentenceTransformer
import pytrec_eval
import logging
from typing import Type, List, Dict, Union, Tuple
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.dense import DenseRetrievalFaissSearch as DRFS
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.sparse import SparseSearch as SS
from beir.retrieval.custom_metrics import mrr, recall_cap, hole, top_k_accuracy


from .AbsTask import AbsTask


logger = logging.getLogger(__name__)

DRES_METHODS = ["encode_queries", "encode_corpus"]
DRPES_METHODS = [
    "start_multi_process_pool",
    "stop_multi_process_pool",
    "encode_queries",
    "encode_corpus",
    "encode_corpus_parallel",
]




class EvaluateRetrieval:
    
    def __init__(self, retriever: Union[Type[DRES], Type[DRFS], Type[BM25], Type[SS]] = None, k_values: List[int] = [1,3,5,10,100,1000], score_function: str = "cos_sim"):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function
            
    def retrieve(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], **kwargs) -> Dict[str, Dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)
    
    def rerank(self, 
            corpus: Dict[str, Dict[str, str]], 
            queries: Dict[str, str],
            results: Dict[str, Dict[str, float]],
            top_k: int) -> Dict[str, Dict[str, float]]:
    
        new_corpus = {}
    
        for query_id in results:
            if len(results[query_id]) > top_k:
                for (doc_id, _) in sorted(results[query_id].items(), key=lambda item: item[1], reverse=True)[:top_k]:
                    new_corpus[doc_id] = corpus[doc_id]
            else:
                for doc_id in results[query_id]:
                    new_corpus[doc_id] = corpus[doc_id]
                    
        return self.retriever.search(new_corpus, queries, top_k, self.score_function)

    @staticmethod
    def evaluate(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int],
                 ignore_identical_ids: bool=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        
        if ignore_identical_ids:
            logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            # _map[f"MAP@{k}"] = 0.0
            # recall[f"Recall@{k}"] = 0.0
            # precision[f"P@{k}"] = 0.0
        
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])

        # map_string = "map_cut." + ",".join([str(k) for k in k_values])
        # recall_string = "recall." + ",".join([str(k) for k in k_values])
        # precision_string = "P." + ",".join([str(k) for k in k_values])
        # evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                # _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                # recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                # precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            # _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            # recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            # precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        for eval in [ndcg, _map, recall, precision]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision
    
    @staticmethod
    def evaluate_custom(qrels: Dict[str, Dict[str, int]], 
                 results: Dict[str, Dict[str, float]], 
                 k_values: List[int], metric: str) -> Tuple[Dict[str, float]]:
        
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)
        
        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)
        
        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)
        
        elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
            return top_k_accuracy(qrels, results, k_values)


class AbsTaskRetrieval(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def is_dres_compatible(model, is_parallel=True):
        methods = DRPES_METHODS if is_parallel else DRES_METHODS
        for method in methods:
            op = getattr(model, method, None)
            if not (callable(op)):
                return False
        return True

    def evaluate(
        self,
        rank, 
        model,
        split="test",
        batch_size=128,
        corpus_chunk_size=None,
        target_devices=None,
        score_function="cos_sim",
        **kwargs
    ):
        # try:
        #     pass
        #     # from beir.retrieval.evaluation import EvaluateRetrieval
        # except ImportError:
        #     raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")

        if not self.data_loaded:
            self.load_data()

        corpus, queries, relevant_docs = self.corpus[split], self.queries[split], self.relevant_docs[split]

        # try:
        #     raise ImportError("MTEB is temporarily incompatible with HFDataLoader")

        #     if self.description["beir_name"].startswith("cqadupstack"):
        #         raise ImportError("CQADupstack is incompatible with latest BEIR")
        #     from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES

        #     model = model if self.is_dres_compatible(model, is_parallel=True) else DRESModel(model)

        #     model = DRPES(
        #         model,
        #         batch_size=batch_size,
        #         target_devices=target_devices,
        #         corpus_chunk_size=corpus_chunk_size,
        #         **kwargs,
        #     )
        # except ImportError:
        #     if target_devices is not None:
        #         logger.warning(
        #             "DenseRetrievalParallelExactSearch could not be imported from beir. Using DenseRetrievalExactSearch instead."
        #         )
        #         logger.warning("The parameter target_devices is ignored.")

        #     from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

        assert self.is_dres_compatible(model, is_parallel=False), "Model must be DRES compatible"
        # model = model if self.is_dres_compatible(model, is_parallel=False) else DRESModel(model)

        model = DRES(
            model,
            batch_size=batch_size,
            corpus_chunk_size=corpus_chunk_size if corpus_chunk_size is not None else 50000,
            **kwargs,
        )

        retriever = EvaluateRetrieval(model, score_function=score_function, k_values=[10])  # or "cos_sim" or "dot"
        start_time = time()
        results = retriever.retrieve(corpus, queries)
        end_time = time()
        print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

        if rank != 0: return {}
        
        ndcg, _map, recall, precision = retriever.evaluate(relevant_docs, results, retriever.k_values)
        # mrr = retriever.evaluate_custom(relevant_docs, results, retriever.k_values, "mrr")

        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            # **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            # **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            # **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            # **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
        }

        return scores


# class DRESModel:
#     """
#     Dense Retrieval Exact Search (DRES) in BeIR requires an encode_queries & encode_corpus method.
#     This class converts a MTEB model (with just an .encode method) into BeIR DRES format.
#     """

#     def __init__(self, model, sep=" ", **kwargs):
#         self.model = model
#         self.sep = sep

#     def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
#         logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

#         ctx = mp.get_context("spawn")
#         input_queue = ctx.Queue()
#         output_queue = ctx.Queue()
#         processes = []

#         for process_id, device_name in enumerate(target_devices):
#             p = ctx.Process(
#                 target=SentenceTransformer._encode_multi_process_worker,
#                 args=(process_id, device_name, self.model, input_queue, output_queue),
#                 daemon=True,
#             )
#             p.start()
#             processes.append(p)

#         return {"input": input_queue, "output": output_queue, "processes": processes}

#     def stop_multi_process_pool(self, pool: Dict[str, object]):
#         output_queue = pool["output"]
#         [output_queue.get() for _ in range(len(pool["processes"]))]
#         return self.model.stop_multi_process_pool(pool)

#     def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
#         return self.model.encode(queries, batch_size=batch_size, **kwargs)

#     def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
#         if type(corpus) is dict:
#             sentences = [
#                 (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
#                 if "title" in corpus
#                 else corpus["text"][i].strip()
#                 for i in range(len(corpus["text"]))
#             ]
#         else:
#             sentences = [
#                 (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
#                 for doc in corpus
#             ]
#         return self.model.encode(sentences, batch_size=batch_size, **kwargs)

#     def encode_corpus_parallel(
#         self, corpus: List[Dict[str, str]], pool: Dict[str, object], batch_size: int, chunk_id: int, **kwargs
#     ):
#         if type(corpus) is dict:
#             sentences = [
#                 (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
#                 if "title" in corpus
#                 else corpus["text"][i].strip()
#                 for i in range(len(corpus["text"]))
#             ]
#         else:
#             sentences = [
#                 (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
#                 for doc in corpus
#             ]

#         if chunk_id is not None and chunk_id >= len(pool["processes"]):
#             output_queue = pool["output"]
#             output_queue.get()

#         input_queue = pool["input"]
#         input_queue.put([chunk_id, batch_size, sentences])
