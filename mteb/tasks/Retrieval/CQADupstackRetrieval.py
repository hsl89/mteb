import os
import datasets 

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from ...abstasks.BeIRTask import BeIRTask
from . import (
    CQADupstackAndroidRetrieval,
    CQADupstackEnglishRetrieval,
    CQADupstackGisRetrieval,
    CQADupstackMathematicaRetrieval,
    CQADupstackProgrammersRetrieval,
    CQADupstackStatsRetrieval,
    CQADupstackUnixRetrieval,
    CQADupstackGamingRetrieval, 
    CQADupstackPhysicsRetrieval, 
    CQADupstackTexRetrieval, 
    CQADupstackWebmastersRetrieval,
    CQADupstackWordpressRetrieval,
)


TASK_LIST = [
    CQADupstackAndroidRetrieval,
    CQADupstackEnglishRetrieval,
    CQADupstackGisRetrieval,
    CQADupstackMathematicaRetrieval,
    CQADupstackProgrammersRetrieval,
    CQADupstackStatsRetrieval,
    CQADupstackUnixRetrieval,
    CQADupstackGamingRetrieval, 
    CQADupstackPhysicsRetrieval, 
    CQADupstackTexRetrieval, 
    CQADupstackWebmastersRetrieval,
    CQADupstackWordpressRetrieval,
]

class CQADupstackRetrieval(AbsTaskRetrieval):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tasks = [t(**kwargs) for t in TASK_LIST]

     
    @property
    def description(self):
        return {
            "name": "CQADupstackRetrieval",
            "beir_name": "cqadupstack",
            "description": "CQADupStack: A Benchmark Data Set for Community Question-Answering Research",
            "reference": "http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "ndcg_at_10",
        }
    
            
    def load_data(self, eval_split=None, **kwargs):
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader as BeirDataLoader
        except ImportError:
            raise Exception("Retrieval tasks require beir package. Please install it with `pip install mteb[beir]`")
        
        dataset = self.description["beir_name"]
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        download_path = os.path.join(datasets.config.HF_DATASETS_CACHE, "BeIR")
        data_path = util.download_and_unzip(url, download_path)
        return  

    def evaluate(self, model, split="test", batch_size=128, corpus_chunk_size=None, 
                 target_device=None, score_function="cos_sim", **kwargs):
        
        average_score = {}
        for task in self.tasks:
            score =  task.evaluate(model, split, batch_size, corpus_chunk_size, target_device, score_function, **kwargs)
            for k, v in score.items():
                average_score[k] = average_score.get(k, 0.0) + (v / len(TASK_LIST)) 
        return average_score