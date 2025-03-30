# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition"""

import os

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{derczynski2016broad,
  title={Broad twitter corpus: A diverse named entity recognition resource},
  author={Derczynski, Leon and Bontcheva, Kalina and Roberts, Ian},
  booktitle={Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers},
  pages={1169--1179},
  year={2016}
}
"""

_DESCRIPTION = """\
This is the Broad Twitter corpus, a dataset of tweets collected over stratified times, places and social uses. 
The goal is to represent a broad range of activities, giving a dataset more representative of the language used 
in this hardest of social media formats to process. Further, the BTC is annotated for named entities.

For more details see [https://aclanthology.org/C16-1111/](https://aclanthology.org/C16-1111/)
"""

_URL = "https://github.com/GateNLP/broad_twitter_corpus/archive/refs/heads/master.zip"
_subpath = "broad_twitter_corpus-master/"
_A_FILE = _subpath + "a.conll"
_B_FILE = _subpath + "b.conll"
_E_FILE = _subpath + "e.conll"
_F_FILE = _subpath + "f.conll"
_G_FILE = _subpath + "g.conll"
_H_FILE = _subpath + "h.conll"

# _TRAINING_FILE = "train.txt"
_DEV_FILE = _H_FILE
_TEST_FILE = _F_FILE


class BroadTwitterCorpusConfig(datasets.BuilderConfig):
    """BuilderConfig for BroadTwitterCorpus"""

    def __init__(self, **kwargs):
        """BuilderConfig for BroadTwitterCorpus.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BroadTwitterCorpusConfig, self).__init__(**kwargs)


class BroadTwitterCorpus(datasets.GeneratorBasedBuilder):
    """BroadTwitterCorpus dataset."""

    BUILDER_CONFIGS = [
        BroadTwitterCorpusConfig(name="broad-twitter-corpus", version=datasets.Version("1.0.0"), description="Broad Twitter Corpus"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-ORG",
                                "I-ORG",
                                "B-LOC",
                                "I-LOC",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://aclanthology.org/C16-1111/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)

        data_files = {
            "a": os.path.join(downloaded_file, _A_FILE),
            "b": os.path.join(downloaded_file, _B_FILE),
            "e": os.path.join(downloaded_file, _E_FILE),
            "f": os.path.join(downloaded_file, _F_FILE),
            "g": os.path.join(downloaded_file, _G_FILE),
            "h": os.path.join(downloaded_file, _H_FILE),
            "dev": os.path.join(downloaded_file, _DEV_FILE),
            "test": os.path.join(downloaded_file, _TEST_FILE),
        }

        """
        btc_section_a = datasets.SplitGenerator(name="BTC_A", gen_kwargs={"filepath": data_files["a"]})
        btc_section_b = datasets.SplitGenerator(name="BTC_B", gen_kwargs={"filepath": data_files["b"]})
        btc_section_e = datasets.SplitGenerator(name="BTC_E", gen_kwargs={"filepath": data_files["e"]})
        btc_section_f = datasets.SplitGenerator(name="BTC_F", gen_kwargs={"filepath": data_files["f"]})
        btc_section_g = datasets.SplitGenerator(name="BTC_G", gen_kwargs={"filepath": data_files["g"]})
        btc_section_h = datasets.SplitGenerator(name="BTC_H", gen_kwargs={"filepath": data_files["h"]})
        """
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": [data_files['a'], data_files['b'], data_files['e'], data_files['g']]}
                ),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": [data_files["dev"]]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": [data_files["test"]]}),
        ]

    def _generate_examples(self, filepaths):
        guid = 0
        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                logger.info("⏳ Generating examples from = %s", filepath)
                tokens = []
                ner_tags = []
                for line in f:
                    if line.startswith("-DOCSTART-") or line.strip() == "" or line == "\n":
                        if tokens:
                            yield guid, {
                                "id": str(guid),
                                "tokens": tokens,
                                "ner_tags": ner_tags,
                            }
                            guid += 1
                            tokens = []
                            ner_tags = []
                    else:
                        # btc entries are tab separated
                        fields = line.split("\t")
                        tokens.append(fields[0])
                        ner_tags.append(fields[1].rstrip())
                # last example
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
                guid += 1 # for when files roll over
