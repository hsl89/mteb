import datasets
from loguru import logger
from .AbsTask import AbsTask


class CrosslingualTask(AbsTask):
    def __init__(self, langs=None, **kwargs):
        super().__init__(**kwargs)
        if type(langs) is list:
            langs = [lang for lang in langs if lang in self.description["eval_langs"]]
        if langs is not None and len(langs) > 0:
            self.langs = langs
        else:
            self.langs = self.description["eval_langs"]
        self.is_crosslingual = True

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return
        self.dataset = {}
        for lang in self.langs:
            try:
                self.dataset[lang] = datasets.load_dataset(
                    self.description["hf_hub_name"],
                    lang,
                    revision=self.description.get("revision", None),
                )
            except ValueError as e:
                logger.warning(e)
    
        if self.dataset == {}:
            raise ValueError("None of the languages %s is present in the dataset: %s" % (str(self.langs, self.description["hf_hub_name"]))) 
        self.data_loaded = True
