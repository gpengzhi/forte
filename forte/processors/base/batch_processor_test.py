# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for BatchProcessor.
"""
import unittest

from typing import Dict, Optional, Type

import numpy as np

from ddt import ddt, data
from texar.torch import HParams

from forte.common.resources import Resources
from forte.common.types import DataRequest
from forte.data.readers import OntonotesReader, StringReader, PlainTextReader
from forte.data.batchers import ProcessingBatcher, FixedSizeDataPackBatcher
from forte.data.data_pack import DataPack
from forte.processors.base import BatchProcessor, FixedSizeBatchProcessor
from forte.processors.nltk_processors import NLTKSentenceSegmenter
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Token, Sentence, EntityMention, RelationLink


class DummyRelationExtractor(BatchProcessor):
    r"""A dummy relation extractor.

    Note that to use :class:`DummyRelationExtractor`, the :attr:`ontology` of
    :class:`Pipeline` must be an ontology that includes
    ``ft.onto.base_ontology.Sentence``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.batcher = self.define_batcher()

    def initialize(self, resource: Resources, configs: Optional[HParams]):
        self.batcher.initialize(configs.batcher)

    def define_batcher(self) -> ProcessingBatcher:
        return FixedSizeDataPackBatcher()

    def define_context(self) -> Type[Sentence]:
        return Sentence

    def _define_input_info(self) -> DataRequest:
        input_info: DataRequest = {
            Token: [],
            EntityMention: {"fields": ["ner_type", "tid"]}
        }
        return input_info

    def predict(self, data_batch: Dict):
        entities_span = data_batch["EntityMention"]["span"]
        entities_tid = data_batch["EntityMention"]["tid"]

        pred: Dict = {
            "RelationLink": {
                "parent.tid": [],
                "child.tid": [],
                "rel_type": [],
            }
        }
        for tid, entity in zip(entities_tid, entities_span):
            parent = []
            child = []
            rel_type = []

            entity_num = len(entity)
            for i in range(entity_num):
                for j in range(i + 1, entity_num):
                    parent.append(tid[i])
                    child.append(tid[j])
                    rel_type.append("dummy_relation")

            pred["RelationLink"]["parent.tid"].append(
                np.array(parent))
            pred["RelationLink"]["child.tid"].append(
                np.array(child))
            pred["RelationLink"]["rel_type"].append(
                np.array(rel_type))

        return pred

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        r"""Add corresponding fields to data_pack"""
        if output_dict is None:
            return

        for i in range(len(output_dict["RelationLink"]["parent.tid"])):
            for j in range(len(output_dict["RelationLink"]["parent.tid"][i])):
                link = RelationLink(data_pack)
                link.rel_type = output_dict["RelationLink"]["rel_type"][i][j]
                parent: EntityMention = data_pack.get_entry(  # type: ignore
                        output_dict["RelationLink"]["parent.tid"][i][j])
                link.set_parent(parent)
                child: EntityMention = data_pack.get_entry(  # type: ignore
                        output_dict["RelationLink"]["child.tid"][i][j])
                link.set_child(child)
                data_pack.add_or_get_entry(link)

    @staticmethod
    def default_configs():
        return {
            "batcher": {"batch_size": 10}
        }


class DummmyFixedSizeBatchProcessor(FixedSizeBatchProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0
        self.batcher = self.define_batcher()

    def initialize(self, resource: Resources, configs: Optional[HParams]):
        self.batcher.initialize(configs.batcher)

    def define_context(self) -> Type[Sentence]:
        return Sentence

    def _define_input_info(self) -> DataRequest:
        return {}

    def predict(self, data_batch: Dict):
        # track the number of times `predict` was called
        self.counter += 1
        return data_batch

    def pack(self, data_pack: DataPack, output_dict: Optional[Dict] = None):
        r"""Add corresponding fields to data_pack"""
        pass

    @staticmethod
    def default_configs():
        return {
            "batcher": {"batch_size": 10}
        }


class DummyProcessorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.nlp = Pipeline()
        self.nlp.set_reader(OntonotesReader())
        dummy = DummyRelationExtractor()
        config = {"batcher": {"batch_size": 5}}
        self.nlp.add_processor(dummy, config=config)
        self.nlp.initialize()

        self.data_path = "data_samples/ontonotes/00/"

    def test_processor(self):
        pack = self.nlp.process(self.data_path)
        relations = list(pack.get_entries(RelationLink))
        assert (len(relations) > 0)
        for relation in relations:
            self.assertEqual(relation.get_field("rel_type"), "dummy_relation")


@ddt
class DummyFixedSizeBatchProcessorTest(unittest.TestCase):

    @data(1, 2, 3)
    def test_one_batch_processor(self, batch_size):
        nlp = Pipeline()
        nlp.set_reader(StringReader())
        dummy = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add_processor(NLTKSentenceSegmenter())
        nlp.add_processor(dummy, config=config)
        nlp.initialize()
        sentences = ["This tool is called Forte. The goal of this project to "
                     "help you build NLP pipelines. NLP has never been made "
                     "this easy before."]
        pack = nlp.process(sentences)
        sent_len = len(list(pack.get(Sentence)))
        self.assertEqual(
            dummy.counter, (sent_len // batch_size +
                            (sent_len % batch_size > 0)))

    @data(1, 2, 3)
    def test_two_batch_processors(self, batch_size):
        nlp = Pipeline()
        nlp.set_reader(PlainTextReader())
        dummy1 = DummmyFixedSizeBatchProcessor()
        dummy2 = DummmyFixedSizeBatchProcessor()
        config = {"batcher": {"batch_size": batch_size}}
        nlp.add_processor(NLTKSentenceSegmenter())

        nlp.add_processor(dummy1, config=config)
        config = {"batcher": {"batch_size": 2 * batch_size}}
        nlp.add_processor(dummy2, config=config)

        nlp.initialize()
        data_path = "data_samples/random_texts"
        pack = nlp.process(data_path)
        sent_len = len(list(pack.get(Sentence)))

        self.assertEqual(
            dummy1.counter, (sent_len // batch_size +
                            (sent_len % batch_size > 0)))

        self.assertEqual(
            dummy2.counter, (sent_len // (2 * batch_size) +
                             (sent_len % (2 * batch_size) > 0)))


if __name__ == '__main__':
    unittest.main()
