# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import logging
import typing as tp
import numpy as np
from pathlib import Path

import fairseq
import torch

from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub
from stopes.eval.alti.wrappers.transformer_wrapper import FairseqMultiEncoderTransformerHub

from .alti_metrics_utils import compute_alti_scores_for_batch
from .file_utils import join_lists_of_dicts, read_tsv, select_columns, write_tsv

logger = logging.getLogger(__name__)

def load_bilingual_model(
    checkpoint: Path,
    data_dir: Path,
    **kwargs,
) -> FairseqTransformerHub:
    """Loading an ALTI hub for a bilingual Fairseq translation model."""
    checkpoint_dir = str(checkpoint.parent)
    checkpoint_filename = str(checkpoint.name)
    alti_hub = FairseqTransformerHub.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_filename,
        data_name_or_path=str(data_dir),
        **kwargs,
    )
    return alti_hub

def load_multi_encoder_model(
    checkpoint: Path,
    data_dir: Path,
    **kwargs,
) -> FairseqMultiEncoderTransformerHub:
    """Loading an ALTI hub for a bilingual Fairseq translation model."""
    checkpoint_dir = str(checkpoint.parent)
    checkpoint_filename = str(checkpoint.name)
    alti_hub = FairseqMultiEncoderTransformerHub.from_pretrained(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_filename,
        data_name_or_path=str(data_dir),
        **kwargs,
    )
    return alti_hub

@dataclasses.dataclass
class ALTIMetricsConfig:
    """The config indicating how to load sentence pairs, load the model,
    compute the ALTI metrics with it, and save results. - to use with the `compute_nllb_alti` function."""

    # the model used to compute ALTI
    is_multilingual: bool
    checkpoint: Path
    data_dir: Path
    #spm: Path
    use_gpu: bool
    # location of the results
    metrics_filename: Path  # a .tsv file with sentence-level metrics
    alignment_filename: tp.Optional[
        Path
    ]  # a .jsonl file with token-level contributions
    # format and location of the source data
    input_filename: Path  # the source file with sources and translations; assumed to be .tsv
    src_lang: str
    tgt_lang: str
    src_col: tp.Union[str, int] = "src"
    tgt_col: tp.Union[str, int] = "mt"
    src_ctx_col: tp.Union[str, int] = "src_ctx"
    tgt_ctx_col: tp.Union[str, int] = "tgt_ctx"
    src_ante_col: tp.Union[str, int] = "src_ante"
    tgt_ante_col: tp.Union[str, int] = "tgt_ante"

def compute_nllb_alti(config: ALTIMetricsConfig) -> None:
    """Compute ALTI+ based attributions and metrics with an NLLB-like models and store the results in files."""
    columns_are_named = any(
        isinstance(c, str) for c in [config.src_col, config.tgt_col, config.src_ctx_col, config.tgt_ctx_col, config.src_ante_col, config.tgt_ante_col]
    )
    input_data = read_tsv(config.input_filename, named_columns=columns_are_named)
    src_texts, tgt_texts, src_contexts, tgt_contexts, src_antecedents, tgt_antecedents = select_columns(input_data, [config.src_col, config.tgt_col, config.src_ctx_col, config.tgt_ctx_col, config.src_ante_col, config.tgt_ante_col])
    if config.is_multi_encoder:
        alti_hub = load_multi_encoder_model(checkpoint=Path(config.checkpoint), data_dir=config.data_dir)
    else:
        alti_hub = load_bilingual_model(checkpoint=Path(config.checkpoint), data_dir=config.data_dir)
    if config.use_gpu:
        if torch.cuda.is_available():
            alti_hub.cuda()
        else:
            logger.warning(
                "You requested use_gpu=True, but there is no GPU available. Falling back to CPU."
            )
    metrics, word_level_metrics, alignments = compute_alti_scores_for_batch(
        alti_hub=alti_hub,
        src_texts=src_texts,
        tgt_texts=tgt_texts,
        src_contexts=src_contexts,
        tgt_contexts=tgt_contexts,
        src_antecedents=src_antecedents,
        tgt_antecedents=tgt_antecedents,
        src_langs=config.src_lang,
        tgt_langs=config.tgt_lang,
    )
    print("antecedent contributions= ", np.mean([i["ante_vs_all"] for i in word_level_metrics]))
    print("context contributions= ", np.mean([i["ctx_vs_current"] for i in metrics]))
    print("current contributions= ", np.mean([i["current_vs_ctx"] for i in metrics]))
    print("source context contributions= ", np.mean([i["src_ctx"] for i in metrics]))
    print("target context contributions= ", np.mean([i["tgt_ctx"] for i in metrics]))
    print("source contributions= ", np.mean([i["src"] for i in metrics]))
    print("target contributions= ", np.mean([i["tgt"] for i in metrics]))
    #write_tsv(config.metrics_filename, join_lists_of_dicts(input_data, metrics))

    #if config.alignment_filename:
    #    with open(config.alignment_filename, "w", encoding="utf-8") as f:
    #        for item in alignments:
    #            json.dump(item, f, ensure_ascii=False)
    #            print(file=f)
