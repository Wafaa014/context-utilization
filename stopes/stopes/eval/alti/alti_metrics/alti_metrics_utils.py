# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import torch
import math
import sentencepiece as sp
from tqdm.auto import trange

from stopes.eval.alti.alignment import align
from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub


def encode(s, spm, vocab):
    """binarizes a sentence according to sentencepiece model and a vocab"""
    tokenized = " ".join(spm.encode(s, out_type=str))
    return vocab.encode_line(tokenized, append_eos=False, add_if_not_exist=False)

def binarize_pair(
    hub: FairseqTransformerHub,
    input_text: str,
    output_text: str,
    input_ctx: str,
    output_ctx: str,
    src_ante: str,
    tgt_ante: str,
    src_lang: tp.Optional[str] = None,
    tgt_lang: tp.Optional[str] = None,
    max_length: tp.Optional[int] = None,
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a pair of texts into source, target and predicted tensors with all special tokens."""

    # st = hub.binarize(hub.apply_bpe(hub.tokenize(input_text)))
    # pt = hub.binarize(hub.apply_bpe(hub.tokenize(output_text)))
    # st_ctx = hub.binarize(hub.apply_bpe(hub.tokenize(input_ctx)))
    # pt_ctx = hub.binarize(hub.apply_bpe(hub.tokenize(output_ctx)))

    src_spm = sp.SentencePieceProcessor()
    tgt_spm = sp.SentencePieceProcessor()
    src_spm.Load(f"{hub.spm_dir}/spm.{src_lang}.model")
    tgt_spm.Load(f"{hub.spm_dir}/spm.{tgt_lang}.model")
    st = encode(input_text, src_spm, hub.src_dict)
    pt = encode(output_text, tgt_spm, hub.tgt_dict)
    st_ctx = encode(input_ctx, src_spm, hub.src_dict)
    pt_ctx = encode(output_ctx, tgt_spm, hub.tgt_dict)    
    src_ante_tensor = encode(src_ante, src_spm, hub.src_dict)
    tgt_ante_tensor = encode(tgt_ante, tgt_spm, hub.tgt_dict)

    st = torch.cat([st, torch.tensor([hub.task.source_dictionary.eos()])])
    st_ctx = torch.cat([st_ctx, torch.tensor([hub.task.source_dictionary.eos()])])
    pt_ctx = torch.cat([pt_ctx, torch.tensor([hub.task.target_dictionary.eos()])])
    tt = pt
    max_length = 200
    if max_length is not None:
        st, pt, tt = st[:max_length], pt[:max_length], tt[:max_length]
        st_ctx, pt_ctx = st_ctx[:max_length], pt_ctx[:max_length]

    return st, pt, st_ctx, pt_ctx, tt, src_ante_tensor, tgt_ante_tensor


def get_loss(
    hub: FairseqTransformerHub,
    input_text: str,
    output_text: str,
    input_ctx: str,
    output_ctx: str,
    src_lang: tp.Optional[str] = None,
    tgt_lang: tp.Optional[str] = None,
) -> tp.Dict[str, float]:
    """Using an ALTI hub, use its model to compute loss for a given text pair."""

    st, pt, st_ctx, pt_ctx, tt = binarize_pair(
        hub, input_text, output_text, input_ctx, output_ctx, src_lang=src_lang, tgt_lang=tgt_lang
    )

    with torch.inference_mode():
        logits, out = hub.models[0].forward(
            src_tokens=st.unsqueeze(0).to(hub.device),
            src_lengths=torch.tensor(st.shape).to(hub.device),
            prev_output_tokens=tt.unsqueeze(0).to(hub.device),
            src_ctx_tokens=st_ctx.unsqueeze(0).to(hub.device),
            src_ctx_lengths=torch.tensor(st_ctx.shape).to(hub.device),
            tgt_ctx_tokens=pt_ctx.unsqueeze(0).to(hub.device),
            tgt_ctx_lengths=torch.tensor(pt_ctx.shape).to(hub.device),
        )
        loss_fct = torch.nn.CrossEntropyLoss()
        log_loss = loss_fct(logits.view(-1, logits.size(-1)), pt.to(hub.device)).item()
    return {"loss_avg": log_loss, "loss_sum": log_loss * len(pt)}


def compute_alti_nllb(
    hub: FairseqTransformerHub,
    input_text: str,
    output_text: str,
    input_ctx: str,
    output_ctx: str,
    src_ante: str,
    tgt_ante: str,
    src_lang: tp.Optional[str] = None,
    tgt_lang: tp.Optional[str] = None,
    max_length: tp.Optional[int] = None,
    contrib_type="l1",
    norm_mode="min_sum",
) -> tp.Tuple[np.ndarray, tp.List[str], tp.List[str], tp.List[str]]:
    """Compute ALTI+ matrix and all tokenized sentences using an NLLB-like seq2seq model."""
    src_tensor, pred_tensor, src_ctx_tensor, tgt_ctx_tensor, tgt_tensor, src_ante_tensor, tgt_ante_tensor = binarize_pair(
        hub,
        input_text,
        output_text,
        input_ctx,
        output_ctx,
        src_ante,
        tgt_ante,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
    )

    source_sentence = hub.decode2(src_tensor, hub.task.source_dictionary)
    target_sentence = hub.decode2(tgt_tensor, hub.task.target_dictionary)
    predicted_sentence = hub.decode2(pred_tensor, hub.task.target_dictionary)
    source_ctx_sentence = hub.decode2(src_ctx_tensor, hub.task.source_dictionary)
    target_ctx_sentence = hub.decode2(tgt_ctx_tensor, hub.task.target_dictionary)
    src_antecedent = hub.decode2(src_ante_tensor, hub.task.source_dictionary)
    tgt_antecedent = hub.decode2(tgt_ante_tensor, hub.task.target_dictionary)
    with torch.inference_mode():
        all_alti = hub.get_contribution_rollout(
            src_tensor, tgt_tensor, src_ctx_tensor, tgt_ctx_tensor, contrib_type=contrib_type, norm_mode=norm_mode
        )
        token_level_alti = all_alti["total"][-1].detach().cpu().numpy()

    return token_level_alti, source_sentence, target_sentence, predicted_sentence, source_ctx_sentence, target_ctx_sentence, src_antecedent, tgt_antecedent


def alti_to_word(
    token_level_alti: np.ndarray,
    source_sentence: tp.List[str],
    target_sentence: tp.List[str],
    predicted_sentence: tp.List[str],
    src_ctx_sentence: tp.List[str],
    tgt_ctx_sentence: tp.List[str],
    src_ante: tp.List[str],
    tgt_ante: tp.List[str],
    eos: str = "</s>",
) -> tp.Tuple[np.ndarray, tp.List[str], tp.List[str], tp.List[str]]:
    """Aggregate token contributions and tokens themselves to the word level."""
    word_level_alti, words_in, words_out = align.contrib_tok2words(
        token_level_alti,
        tokens_in=src_ctx_sentence +  source_sentence + tgt_ctx_sentence + target_sentence,
        tokens_out=predicted_sentence,
    )
    src_ante_words, tgt_ante_words = align.tok2words(src_ante, tgt_ante)
    return word_level_alti, words_in, words_out, src_ante_words, tgt_ante_words 


def entropy(proba: np.ndarray) -> float:
    """Literally the formula of entropy"""
    return np.sum(proba * -np.log(proba))


def compute_alti_metrics(
    alti: np.ndarray,
    source_sentence: tp.List[str],
    target_sentence: tp.List[str],
    predicted_sentence: tp.List[str],
    src_ctx_sentence: tp.List[str],
    tgt_ctx_sentence: tp.List[str],
    skip_first: bool = False,
) -> tp.Dict[str, float]:
    """Compute sentence-level metrics of the alignment quality based on the contributions matrix."""
    # for each of the metrics, the higher it is, the better is the alignment
    sc_ctx = alti[:, : len(src_ctx_sentence)]
    sc = alti[:, len(src_ctx_sentence):len(src_ctx_sentence)+len(source_sentence)]
    source_length = len(src_ctx_sentence)+len(source_sentence)
    tgt_ctx = alti[:, source_length: source_length+len(tgt_ctx_sentence)] 
    tgt = alti[:, source_length+len(tgt_ctx_sentence):]
    #if skip_first:  # skip first token if it is special (e.g. language token after BOS)
    #    sc = sc[1:, :]
    src_ax, tgt_ax = 0, 1
    total_sc_ctx = sc_ctx.sum(tgt_ax).mean()
    total_sc = sc.sum(tgt_ax).mean()
    total_tgt_ctx = tgt_ctx.sum(tgt_ax).mean()
    total_tgt = tgt.sum(tgt_ax).mean()
    total_source_sum = total_sc + total_sc_ctx
    total_target_sum = total_tgt + total_tgt_ctx
    total_ctx_sum = total_sc_ctx + total_tgt_ctx
    total_current_sum = total_sc + total_tgt
    total = total_source_sum + total_target_sum
    return dict(
        ctx_vs_current = (total_ctx_sum)*100/(total),
        current_vs_ctx = (total_current_sum)*100/(total),
        src_ctx_vs_src = (total_sc_ctx)*100/(total_source_sum),
        tgt_ctx_vs_tgt = (total_tgt_ctx)*100/(total_target_sum),
        src_ctx = (total_sc_ctx)*100/(total),
        tgt_ctx = (total_tgt_ctx)*100/(total),
        src = (total_sc)*100/(total),
        tgt = (total_tgt)*100/(total),
    )


def compute_alti_metrics_word_level(
    word_level_alti: np.ndarray,
    words_in: tp.List[str],
    words_out: tp.List[str],
    src_ante_words: tp.List[str],
    tgt_ante_words: tp.List[str],
) -> tp.Dict[str, float]:
    """Compute sentence-level metrics of the alignment quality based on the contributions matrix."""
    # for each of the metrics, the higher it is, the better is the alignment
    ante = []

    for word in src_ante_words:
        if word in words_in:
            index = words_in.index(word)
            ante.append(word_level_alti[:, index])
        elif word+"'s" in words_in:
            index = words_in.index(word+ "'s")
            ante.append(word_level_alti[:, index])
        elif word+"'t" in words_in:
            index = words_in.index(word+ "'t")
            ante.append(word_level_alti[:, index])
        elif word+"'re" in words_in:
            index = words_in.index(word+ "'re")
            ante.append(word_level_alti[:, index])
        elif word+"''d" in words_in:
            index = words_in.index(word+ "'d")
            ante.append(word_level_alti[:, index])
        elif word+"'ll" in words_in:
            index = words_in.index(word+ "'ll")
            ante.append(word_level_alti[:, index])

    if tgt_ante_words != ['None']:
        for word in tgt_ante_words:
            if word in words_in:
                index = words_in.index(word)
                ante.append(word_level_alti[:, index])
            elif word+"'s" in words_in:
                index = words_in.index(word+ "'s")
                ante.append(word_level_alti[:, index])
            elif word+"'t" in words_in:
                index = words_in.index(word+ "'t")
                ante.append(word_level_alti[:, index])
            elif word+"'re" in words_in:
                index = words_in.index(word+ "'re")
                ante.append(word_level_alti[:, index])
            elif word+"'d" in words_in:
                index = words_in.index(word+ "'d")
                ante.append(word_level_alti[:, index])
            elif word+"'ll" in words_in:
                index = words_in.index(word+ "'ll")
                ante.append(word_level_alti[:, index])
            elif "l'" + word in words_in:
                index = words_in.index("l'" + word)
                ante.append(word_level_alti[:, index])
            elif "L'" + word in words_in:
                index = words_in.index("L'" + word)
                ante.append(word_level_alti[:, index])
            elif "d'" + word in words_in:
                index = words_in.index("d'" + word)
                ante.append(word_level_alti[:, index])
    src_ax, tgt_ax = 0, 1
    ante = np.array(ante)
    if ante.shape == (0,):
    	ante_sum = 0
    else:
        ante_sum = np.sum(ante, axis=1).mean()
    total = word_level_alti.sum(tgt_ax).mean()
    return dict(ante_vs_all = (ante_sum)*100/(total),
               ante = ante_sum,
               tot = total)


def compute_alti_scores_for_batch(
    alti_hub: FairseqTransformerHub,
    src_texts: tp.List[str],
    tgt_texts: tp.List[str],
    src_contexts: tp.List[str],
    tgt_contexts: tp.List[str],
    src_antecedents: tp.List[str],
    tgt_antecedents: tp.List[str],
    src_langs: tp.Union[None, str, tp.List[str]] = None,
    tgt_langs: tp.Union[None, str, tp.List[str]] = None,
    alignment_threshold: float = 0.08,
) -> tp.Tuple[tp.List[tp.Dict[str, float]], tp.List[tp.Dict]]:
    """Compute ALTI, sentence-level metrics and alignments for a list of sentence pairs."""
    results = []
    word_level_results = []
    alignments = []
    for i in trange(len(src_texts)):
        src_lang = src_langs[i] if isinstance(src_langs, list) else src_langs
        tgt_lang = tgt_langs[i] if isinstance(tgt_langs, list) else tgt_langs
        token_level_alti, src_toks, tgt_toks, pred_toks, src_ctx_toks, tgt_ctx_toks, src_ante_toks, tgt_ante_toks = compute_alti_nllb(
            alti_hub, src_texts[i], tgt_texts[i], src_contexts[i], tgt_contexts[i], src_antecedents[i], tgt_antecedents[i], src_lang=src_lang, tgt_lang=tgt_lang
        )

        word_level_alti, words_in, words_out, src_ante_words, tgt_ante_words = alti_to_word(token_level_alti, src_toks, tgt_toks, pred_toks, src_ctx_toks, tgt_ctx_toks, src_ante_toks, tgt_ante_toks)
        
        metrics = compute_alti_metrics(token_level_alti, src_toks, tgt_toks, pred_toks, src_ctx_toks, tgt_ctx_toks)
        word_level_metrics = compute_alti_metrics_word_level(word_level_alti, words_in, words_out, src_ante_words, tgt_ante_words)
        if any(math.isnan(value) for value in metrics.values()):
            pass
        else:
            results.append(metrics)
        if any(math.isnan(value) for value in word_level_metrics.values()):
            pass
        else:
            word_level_results.append(word_level_metrics)
        # TODO: find a better alignment algorithm
        alignment = [
            (int(x), int(y))
            for x, y in zip(*np.where(token_level_alti > alignment_threshold))
        ]
        alignments.append(
            {
                "contributions": token_level_alti.tolist(),
                "alignment": alignment,
                "src_toks": src_toks,
                "tgt_toks": tgt_toks,
                "pred_toks": pred_toks,
            }
        )
    return results, word_level_results, alignments
