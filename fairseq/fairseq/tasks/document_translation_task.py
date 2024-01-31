from argparse import Namespace
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)
from fairseq.data import indexed_dataset, data_utils, encoders

import os
import json
from dataclasses import dataclass, field

from contextual_mt import ContextualDataset, ContextualSequenceGenerator


@dataclass
class ContextualTranslationConfig(TranslationConfig):
    source_context_size: int = field(
        default=0, metadata={"help": "number of previous source sentences/messages to include in the context"}
    )

    target_context_size: int = field(
        default=0, metadata={"help": "number of previous target sentences/messages to include in the context"}
    )

    sample_context_size: bool = field(
        default=False, metadata={"help": "sample context size"}
    )

    break_tag: str = field(
        default="<brk>", metadata={"help": "is set, separates context sentences by the break tag"}
    )

@register_task("document_translation", dataclass=ContextualTranslationConfig)
class DocumentTranslationTask(TranslationTask):

    cfg: ContextualTranslationConfig

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            assert self.cfg.eval_bleu_detok is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args), seq_gen_cls=ContextualSequenceGenerator
            )
        return model

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(
                data_path, "{}.{}-{}.{}".format(split, src, tgt, lang)
            )
            return indexed_dataset.dataset_exists(filename, impl=self.cfg.dataset_impl)

        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        if split_exists(split, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, src, tgt))
        elif split_exists(split, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split, tgt, src))
        else:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, data_path)
            )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, self.src_dict, self.cfg.dataset_impl
        )
        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, self.tgt_dict, self.cfg.dataset_impl
        )
        with open(prefix + "docids", "r") as f:
            doc_ids = [int(idx) for idx in f]


        self.datasets[split] = ContextualDataset(
            src_dataset,
            src_dataset.sizes,
            self.src_dict,
            tgt_dataset,
            tgt_dataset.sizes,
            self.tgt_dict,
            doc_ids,
            self.cfg.source_context_size,
            self.cfg.target_context_size,
            break_tag=self.cfg.break_tag,
            sample_context_size=self.cfg.sample_context_size,
            shuffle=True,
        )
