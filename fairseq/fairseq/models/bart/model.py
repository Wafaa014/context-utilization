# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BART: Denoising Sequence-to-Sequence Pre-training for
Natural Language Generation, Translation, and Comprehension
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import BARTHubInterface

logger = logging.getLogger(__name__)

# new imports begin here
from typing import Optional, List, Dict, Any
from torch import Tensor
from fairseq.modules import LayerDropModuleList, TransformerEncoderLayer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)


# new imports end here

@register_model("bart")
class BARTModel(TransformerModel):
    __jit_unused_properties__ = ["supported_targets"]

    @classmethod
    def hub_models(cls):
        return {
            "bart.base": "http://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz",
            "bart.large": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz",
            "bart.large.mnli": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz",
            "bart.large.cnn": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz",
            "bart.large.xsum": "http://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz",
        }

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @staticmethod
    def add_args(parser):
        super(BARTModel, BARTModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            help="Apply spectral normalization on the classification head",
        )
        parser.add_argument(
            "--context-loss",
            default=False,
            action="store_true",
            help="if set, trains to predict target context tokens",
        )
        parser.add_argument(
            "--tgt-coword-dropout",
            default=0.0,
            type=float,
            help="if set to value>0, randomly drops target tokens",
        )
        parser.add_argument(
            "--src-coword-dropout",
            default=0.0,
            type=float,
            help="if set to value>0, randomly drops source tokens",
        )
        parser.add_argument(
            "--coword-dropout-type",
            choices=("sample", "predefined_sample", "whole", "suffix"),
            default="sample",
            help="type of coword dropout to use. NOTE: only sample is used"
                 "used in the paper",
        )
        parser.add_argument(
            "--multi-encoder",
            default=False,
            action="store_true",
            help="wether to use multi-encoder in the source side",
        )

    @property
    def supported_targets(self):
        return {"self"}

    @classmethod
    def from_pretrained(
            cls,
            model_name_or_path,
            checkpoint_file="model.pt",
            data_name_or_path=".",
            bpe="gpt2",
            sample_break_mode="eos",
            **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])

    
    def register_classification_head(
            self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        logger.info("Registering classification head: {0}".format(name))
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = BARTClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            do_spectral_norm=getattr(
                self.args, "spectral_norm_classification_head", False
            ),
        )

    def load_state_dict(
            self,
            state_dict,
            strict=False,
            model_cfg=None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        model_dict = self.state_dict()
        # filter out keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        self.upgrade_state_dict(state_dict)
        # here allow partial loading from other models
        return super().load_state_dict(state_dict, strict=False)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
                ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
                ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes
                        != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim
                        != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
                loaded_dict_size == len(self.encoder.dictionary) + 1
                and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
                self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                                          -1, :
                                          ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                    : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                    : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

    def set_beam_size(self, beam):
        """Set beam size for efficient beamable enc-dec attention."""
        beamable = False
        for layer in self.decoder.layers:
            if layer.encoder_attn is not None:
                if hasattr(layer.encoder_attn, "set_beam_size"):
                    layer.encoder_attn.set_beam_size(beam)
                    beamable = True
        if beamable:
            self.encoder.reorder_encoder_out = self.encoder._reorder_encoder_out

            # deleted the forward function of mBART

    # newly added code starts here
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ContextualTransformerEncoder(
            args,
            src_dict,
            embed_tokens,
            multi_encoder=getattr(args, "multi_encoder", False),
            coword_dropout_prob=getattr(args, "src_coword_dropout", 0.0),
            coword_dropout_type=getattr(args, "coword_dropout_type", "sample"),
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ContextualTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            multi_encoder=getattr(args, "multi_encoder", False),
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            coword_dropout_prob=getattr(args, "tgt_coword_dropout", 0.0),
            coword_dropout_type=getattr(args, "coword_dropout_type", "sample"),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            src_ctx_tokens=None,
            src_ctx_lengths=None,
            tgt_ctx_tokens=None,
            tgt_ctx_lengths=None,
            src_sample_probs=None,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            src_ctx_tokens=src_ctx_tokens,
            src_ctx_lengths=src_ctx_lengths,
            src_sample_probs=src_sample_probs,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            context_tokens=tgt_ctx_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out #, encoder_out


class ContextualTransformerEncoder(TransformerEncoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            multi_encoder=False,
            coword_dropout_type="sample",
            coword_dropout_prob=0.0,
    ):
        super().__init__(args, dictionary, embed_tokens)
        self.coword_dropout_type = coword_dropout_type
        self.coword_dropout_prob = coword_dropout_prob
        # TODO: add this a variable token
        self.mask_id = dictionary.index("<mask>")
        self.multi_encoder = multi_encoder
        if self.multi_encoder:
            if self.encoder_layerdrop > 0.0:
                self.context_layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.context_layers = nn.ModuleList([])

            self.context_layers.extend(
                [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
            )

        self.num_layers = len(self.layers)

    def forward(
            self,
            src_tokens,
            src_lengths,
            src_ctx_tokens,
            src_ctx_lengths,
            src_sample_probs=None,
            return_all_hiddens: bool = False,
    ):
        # if source dropout enabled, randomly drop tokens from input
        if self.training and self.coword_dropout_type is not None:
            if self.coword_dropout_type == "sample":
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                probs = torch.ones_like(src_tokens) * self.coword_dropout_prob
                mask = torch.logical_and(
                    torch.bernoulli(probs), torch.logical_not(padding_mask)
                )
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.coword_dropout_type == "predefined_sample":
                # This is used for sampling with token specific probabilies
                # NOTE: this was not used in the paper
                assert (
                        src_sample_probs is not None
                ), "need sample probabilities as a given"
                padding_mask = src_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(src_tokens)
                mask = torch.logical_and(
                    torch.bernoulli(src_sample_probs), torch.logical_not(padding_mask)
                )
                src_tokens = torch.where(mask == 0, src_tokens, mask_token)
            elif self.coword_dropout_type == "whole":
                # make tensor with a single token (mask token)
                # NOTE: not used in the paper
                mask_samples = torch.zeros_like(src_tokens).to(src_tokens)
                mask_samples[mask_samples == 0] = self.padding_idx
                mask_samples[:, 0] = self.mask_id
                # replace samples by this tensor based on bernoulli
                probs = torch.ones((src_tokens.size(0),)) * self.coword_dropout_prob
                mask = torch.bernoulli(probs).to(src_tokens)
                mask = torch.unsqueeze(mask, -1).repeat(1, src_tokens.size(1))
                src_tokens = torch.where(mask == 0, src_tokens, mask_samples)
            else:
                raise ValueError(
                    f"unknown type of source dropout {self.coword_dropout_type}"
                )

        # Encode source tokens
        # as simple context encoding, we just concatenate context to input
        # TODO: add option for separate encoder
        # how to do it so that input can still attend to context
        def encode(tokens, layers):
            padding_mask = tokens.eq(self.padding_idx)
            x, encoder_embedding = self.forward_embedding(tokens)
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            x_encoder_states = []
            for layer in layers:
                x = layer(x, padding_mask)
                if return_all_hiddens:
                    assert x_encoder_states is not None
                    x_encoder_states.append(x)

            if self.layer_norm is not None:
                x = self.layer_norm(x)

            return x, padding_mask, encoder_embedding, x_encoder_states

        if self.multi_encoder:
            ctx_x, ctx_padding_mask, ctx_enc_embeddings, ctx_x_enc_states = encode(
                src_ctx_tokens, self.context_layers
            )
            x, padding_mask, encoder_embedding, x_encoder_states = encode(
                src_tokens, self.layers
            )

            x = torch.cat([ctx_x, x], axis=0)
            padding_mask = torch.cat([ctx_padding_mask, padding_mask], axis=1)
            encoder_embedding = torch.cat(
                [ctx_enc_embeddings, encoder_embedding], axis=1
            )
            x_encoder_states = [
                torch.cat([ctx_states, states], axis=0)
                for ctx_states, states in zip(ctx_x_enc_states, x_encoder_states)
            ]

        else:
            x, padding_mask, encoder_embedding, x_encoder_states = encode(
                torch.cat([src_ctx_tokens, src_tokens], axis=1), self.layers
            )

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": x_encoder_states,  # List[T x B x C]
            "src_tokens": torch.empty(0),
            "src_lengths": torch.empty(0),
        }


class ContextualTransformerDecoder(TransformerDecoder):
    def __init__(
            self,
            args,
            dictionary,
            embed_tokens,
            multi_encoder=False,
            no_encoder_attn=False,
            coword_dropout_type="sample",
            coword_dropout_prob=0.0,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.coword_dropout_type = coword_dropout_type
        self.coword_dropout_prob = coword_dropout_prob
        self.mask_id = dictionary.index("<mask>")

        self.multi_encoder = multi_encoder
        #self.beam = getattr(args, "beam", 1),
        if self.multi_encoder:
            if self.decoder_layerdrop > 0.0:
                self.context_layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.context_layers = nn.ModuleList([])

            self.context_layers.extend(
                [self.build_encoder_layer(args) for i in range(args.decoder_layers)]
            )

    def build_encoder_layer(self, args):
        layer = TransformerEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward_embedding(self, tokens, token_embedding: Optional[torch.Tensor] = None):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
            self,
            prev_output_tokens,
            context_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            context_tokens (LongTensor): context tokens (ie a prefix
                to prev_output_tokens), shape `(batch, tgt_ctx_len)`
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            context_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
            self,
            prev_output_tokens,
            context_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            context_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            context_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = 0  # self.num_layers - 1

        # if target context dropout enabled, randomly drop tokens from input
        if self.training and self.coword_dropout_type is not None:
            if self.coword_dropout_type == "sample":
                padding_mask = context_tokens.eq(self.padding_idx)
                mask_token = torch.tensor(self.mask_id).to(context_tokens)
                probs = torch.ones_like(context_tokens) * self.coword_dropout_prob
                mask = torch.logical_and(
                    torch.bernoulli(probs), torch.logical_not(padding_mask)
                )
                context_tokens = torch.where(mask == 0, context_tokens, mask_token)
            else:
                raise ValueError(
                    f"unknown type of target context dropout {self.coword_dropout_type}"
                )

        if self.multi_encoder:
            #if self.beam[0] > 1:
            #print(context_tokens[0].size())
            ctx_padding_mask = context_tokens.eq(self.padding_idx)
            ctx_x, _ = self.forward_embedding(context_tokens)
            # B x T x C -> T x B x C
            ctx_x = ctx_x.transpose(0, 1)
            for layer in self.context_layers:
                ctx_x = layer(ctx_x, ctx_padding_mask)

            if self.layer_norm is not None:
                ctx_x = self.layer_norm(ctx_x)

            #ctx_x = ctx_x[:,0,:]
            #ctx_x = torch.unsqueeze(ctx_x, 0)
            #ctx_padding_mask = ctx_padding_mask[0]
            #ctx_padding_mask = torch.unsqueeze(ctx_padding_mask, 0)
            #print(ctx_padding_mask.size())
            input_tokens = prev_output_tokens
        else:
            input_tokens = torch.cat([context_tokens, prev_output_tokens], axis=1)
            context_end_id = context_tokens.size(1)

        # embed positions
        if self.embed_positions is not None:
            # concat context_tokens to input
            # FIXME: this is really simple
            positions = self.embed_positions(
                input_tokens, incremental_state=incremental_state
            )
        else:
            positions = None

        if incremental_state is not None and len(incremental_state) > 0:
            input_tokens = input_tokens[:, -1:]
            context_end_id = 0
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(input_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or input_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = input_tokens.eq(self.padding_idx)

        if self.multi_encoder:
            #print("context: ", ctx_x[:,0,:].size())
            #print("beam: ", self.beam, type(self.beam[0]), self.beam[0] == 1)
            #print(encoder_out["encoder_out"][0].size())
            cross_attn = (
                torch.cat([encoder_out["encoder_out"][0], ctx_x], axis=0)
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else ctx_x
            )
            cross_attn_mask = (
                torch.cat(
                    [encoder_out["encoder_padding_mask"][0], ctx_padding_mask], axis=1
                )
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else ctx_padding_mask
            )
        else:
            cross_attn = (
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None
            )
            cross_attn_mask = (
                encoder_out["encoder_padding_mask"][0]
                if (
                        encoder_out is not None
                        and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None
            )
        #print("size of x: ", x.size())
        #print("size of cross attention: ", cross_attn.size())
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if (
                    incremental_state is None or len(incremental_state) == 0
            ) and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            x, layer_attn, _ = layer(
                x,
                cross_attn,
                cross_attn_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            if attn.dim() == 4:
                attn = attn.mean(dim=0)
        # logger.info("I am here Fofi, not removing context")
        # remove context
        if not self.multi_encoder:
            x = x[context_end_id:]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    # newly added code ends here


class BARTClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            activation_fn,
            pooler_dropout,
            do_spectral_norm=False,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

        if do_spectral_norm:
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("bart", "bart_large")
def bart_large_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)


@register_model_architecture("bart", "bart_base")
def bart_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    bart_large_architecture(args)


@register_model_architecture("bart", "mbart_large")
def mbart_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_large_architecture(args)


@register_model_architecture("bart", "mbart_base")
def mbart_base_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    bart_base_architecture(args)


@register_model_architecture("bart", "mbart_base_wmt20")
def mbart_base_wmt20_architecture(args):
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    mbart_base_architecture(args)

