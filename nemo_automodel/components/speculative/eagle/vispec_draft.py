# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""ViSpec (vision-aware) draft model for speculative decoding on VLM targets.

ViSpec (Kang et al., NeurIPS 2025, arXiv:2509.15235) extends the EAGLE-1/2
draft with two vision-specific modules, keeping the rest of the draft
(``embed_tokens`` / ``fc`` / decoder layers / ``norm``) byte-identical to
:class:`~nemo_automodel.components.speculative.eagle.draft_llama_v12.LlamaEagleDraftModel`
so a text-only EAGLE-1/2 checkpoint can initialize stage-2 ViSpec training:

* :class:`VispecImageAdaptor` -- ``num_query_tokens`` learnable queries
  cross-attend over the target's image-token features and compress a whole
  image span (hundreds to thousands of tokens) into ``num_query_tokens``
  vectors. ``num_query_tokens - 1`` of them are spliced back into the draft
  sequence at the *original* trailing positions of the image span, so the
  positional layout of the surrounding text is untouched.
* ``img_fc`` -- the remaining ("global") image vector is broadcast onto every
  subsequent text position and mixed in with a ``[2*hidden -> hidden]``
  projection, giving each text token vision context without paying for image
  tokens in the draft's KV cache.

Reference implementation: ``vispec/model/cnets_ours.py`` in
https://github.com/KangJialiang/ViSpec.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from nemo_automodel.components.speculative.eagle.draft_llama_v12 import (
    LlamaEagleDraftModel,
    _build_causal_mask,
    resolve_attention_bias,
    resolve_fc_bias,
)
from nemo_automodel.components.speculative.eagle.msd_decode import (
    MSDTreeNode,
    MSDTreeProposal,
    build_msd_tree_layout,
)

_DraftKVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class _VispecDraftCandidate:
    """One candidate in the cached draft lattice.

    Attributes:
        candidate_index: Consecutive index in generation order.
        parent_candidate_index: Parent index, or -1 for a root child.
        token_id: Draft token vocabulary id.
        parent_hidden: Tensor of shape [hidden] predicted at the parent.
        log_probability: Cumulative root-to-node draft log probability.
        ancestor_cache_indices: Physical draft-cache indices of tree ancestors.
    """

    candidate_index: int
    parent_candidate_index: int
    token_id: int
    parent_hidden: torch.Tensor
    log_probability: float
    ancestor_cache_indices: tuple[int, ...]


def apply_vispec_draft_architecture(config: PretrainedConfig) -> None:
    """Pin a target-derived draft config to ViSpec's released draft architecture.

    Both ViSpec stages derive the draft config from the target's text config, so
    without this the draft silently inherits whatever the target's language tower
    happens to use. The released draft is not that: ``qwen2.5_vl_7B_config.json``
    in the reference repository sets ``num_attention_heads: 28``,
    ``num_key_value_heads: 28`` and ``qkv_bias: true``, whereas the matching
    Qwen2.5-VL-7B text tower is 28-head GQA over 4 KV heads. Copying the target
    therefore produced a materially smaller draft than the paper's, which is not
    a configuration difference a reader of the recipe would notice.

    These settings cannot be read off the target: HF's ``Qwen2_5_VLTextConfig``
    exposes neither ``attention_bias`` nor ``qkv_bias`` (its attention hard-codes
    the qkv bias inside the module), so there is nothing to inherit. They are
    properties of the ViSpec draft rather than of any one target, and are applied
    unconditionally for every ViSpec target.

    Args:
        config: The draft config, already derived from the target's text config.
            Modified in place.
    """
    # Full multi-head attention, not the target's GQA.
    config.num_key_value_heads = config.num_attention_heads
    # Bias on q/k/v (and the adaptor's k/v); ``o_proj`` stays bias-free.
    config.qkv_bias = True
    # Bias on ``fc`` and ``img_fc``, matching the reference's ``bias=True`` default.
    config.fc_bias = True
    # The reference skips input normalization on layer 0 and has no final norm.
    config.draft_skip_first_input_norm = True
    config.draft_apply_final_norm = False
    # Its attention path uses PyTorch SDPA rather than the common draft's
    # explicit FP32-softmax implementation.
    config.draft_use_sdpa_attention = True


class VispecImageAdaptor(nn.Module):
    """Compress an image-token span into a small set of learnable-query vectors.

    A single non-causal cross-attention: ``num_query_tokens`` learnable queries
    attend over the image-token features, so the output length is independent
    of how many image tokens the target emitted.

    Args:
        config: Draft config supplying ``hidden_size`` and ``num_attention_heads``.
        num_query_tokens: Number of learnable queries (ViSpec's ``num_q``).
    """

    def __init__(self, config: PretrainedConfig, num_query_tokens: int):
        super().__init__()
        if num_query_tokens < 2:
            # One query would leave nothing to splice back into the sequence
            # after the global vector is taken (see ``VispecDraftModel``).
            raise ValueError(f"vispec_num_query_tokens must be >= 2, got {num_query_tokens}")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # ViSpec derives the adaptor head dim from hidden_size (not config.head_dim):
        # the adaptor is its own module, not a copy of the target's attention.
        self.head_dim = self.hidden_size // self.num_heads
        self.num_query_tokens = num_query_tokens

        self.query = nn.Parameter(torch.empty(num_query_tokens, self.num_heads, self.head_dim))
        # The reference's ``ImgAdaptor`` splits bias exactly like the draft's own
        # attention, so it resolves through the same helper rather than restating
        # the rule: k/v follow the qkv convention, ``o_proj`` stays bias-free.
        kv_bias, _ = resolve_attention_bias(config)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=kv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=kv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def reset_query(self) -> None:
        """Re-draw the learnable queries from ``N(0, head_dim ** -0.5)`` (ViSpec's init)."""
        nn.init.normal_(self.query, mean=0.0, std=self.head_dim**-0.5)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Compress image features into ``num_query_tokens`` vectors.

        Args:
            image_features: Tensor of shape [batch, image_tokens, hidden] holding
                the target's embedding-layer output at the image positions of one
                image span.

        Returns:
            Tensor of shape [batch, num_query_tokens, hidden].
        """
        batch_size, seq_len, _ = image_features.shape
        query = self.query.view(1, self.num_query_tokens, self.num_heads, self.head_dim)
        query = query.to(image_features.dtype).transpose(1, 2).expand(batch_size, -1, -1, -1)
        key = self.k_proj(image_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(image_features).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query.contiguous(), key.contiguous(), value.contiguous(), is_causal=False
        )
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, self.num_query_tokens, self.hidden_size)
        return self.o_proj(attn_output)


class VispecDraftModel(LlamaEagleDraftModel):
    """EAGLE-1/2 draft extended with ViSpec's image compression and global image feature.

    Adds two config fields on top of the EAGLE-1/2 draft config:

    * ``vispec_num_query_tokens`` (default 2) -- queries per image span.
    * ``draft_num_hidden_layers`` -- inherited, unchanged.

    The base draft's parameters (``embed_tokens``, ``fc``, ``layers``, ``norm``)
    keep their names, so ``load_state_dict(..., strict=False)`` restores a
    stage-1 EAGLE-1/2 checkpoint and leaves only ``img_adaptor`` / ``img_fc``
    freshly initialized.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_query_tokens = int(getattr(config, "vispec_num_query_tokens", 2))
        self.img_adaptor = VispecImageAdaptor(config, self.num_query_tokens)
        # ``img_fc`` follows the base draft's ``fc``, which is how the reference
        # builds them (both take its single ``bias`` argument).
        self.img_fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=resolve_fc_bias(config))
        self.reset_vispec_parameters()

    def reset_vispec_parameters(self) -> None:
        """Initialize the ViSpec-only modules (identity-start for ``img_fc``).

        ``img_fc`` starts as ``[I | 0]``: it copies the target hidden state
        through and ignores the global image vector, so a stage-2 run
        initialized from a stage-1 EAGLE-1/2 checkpoint starts numerically
        equal to that checkpoint and only then learns to use vision context.
        """
        self.img_adaptor.reset_query()
        hidden_size = self.img_fc.weight.shape[0]
        with torch.no_grad():
            nn.init.eye_(self.img_fc.weight[:, :hidden_size])
            nn.init.zeros_(self.img_fc.weight[:, hidden_size:])
            if self.img_fc.bias is not None:
                nn.init.zeros_(self.img_fc.bias)

    def _fuse(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        global_image_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the global image vector into the target hidden state, then run EAGLE's ``fc``.

        Args:
            inputs_embeds: Tensor of shape [tokens, hidden].
            target_hidden_states: Tensor of shape [tokens, hidden].
            global_image_feature: Tensor of shape [1, hidden], broadcast over ``tokens``.

        Returns:
            Tensor of shape [tokens, hidden].
        """
        vision_context = global_image_feature.expand_as(target_hidden_states)
        hidden_states = self.img_fc(torch.cat((target_hidden_states, vision_context), dim=-1))
        return self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

    def _compress_sequence(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the compressed draft sequence for one (batch-size-1) sample.

        Walks the sample image span by image span. Each span contributes its
        leading text positions (fused with the *previous* span's global image
        vector) followed by ``num_query_tokens - 1`` compressed image tokens;
        the trailing text after the last span is fused with the last span's
        global vector. Because every span ends with its image run, this
        reordering preserves the original left-to-right token order.

        Args:
            inputs_embeds: Tensor of shape [sequence, hidden].
            target_hidden_states: Tensor of shape [sequence, hidden].
            image_mask: Bool tensor of shape [sequence]; True at image positions.

        Returns:
            Tuple of ``(hidden_states, source_index, global_image_feature)``:
            ``hidden_states`` is a Tensor of shape [compressed_sequence, hidden];
            ``source_index`` is a long Tensor of shape [compressed_sequence]
            giving, for each compressed position, its position in the original
            sequence; ``global_image_feature`` is a Tensor of shape [1, hidden]
            retained for subsequent text-only cached decoding.
        """
        seq_len = inputs_embeds.shape[0]
        positions = torch.arange(seq_len, device=inputs_embeds.device)
        # One entry per contiguous image run: the index just past its last token.
        span_ends = torch.cat((image_mask[:-1] & ~image_mask[1:], image_mask[-1:]))
        span_end_ids = torch.nonzero(span_ends, as_tuple=False).flatten() + 1

        num_spliced = self.num_query_tokens - 1
        global_image_feature = torch.zeros_like(inputs_embeds[:1])
        segments: list[torch.Tensor] = []
        source_index: list[torch.Tensor] = []
        span_start = 0
        for span_end in span_end_ids.tolist():
            span_image_mask = image_mask[span_start:span_end]
            text_index = positions[span_start:span_end][~span_image_mask]
            segments.append(
                self._fuse(
                    inputs_embeds[text_index],
                    target_hidden_states[text_index],
                    global_image_feature,
                )
            )
            source_index.append(text_index)

            image_index = positions[span_start:span_end][span_image_mask]
            if image_index.numel() < num_spliced:
                raise ValueError(
                    f"ViSpec splices {num_spliced} compressed tokens back into the positions of each image "
                    f"span, but this span holds only {int(image_index.numel())} image token(s). Lower "
                    "vispec_num_query_tokens or raise the processor's image resolution."
                )
            compressed = self.img_adaptor(inputs_embeds[image_index].unsqueeze(0)).squeeze(0)
            segments.append(compressed[:num_spliced])
            # The spliced tokens inherit the span's trailing original positions,
            # so the surrounding text keeps its own position ids untouched.
            source_index.append(positions[span_end - num_spliced : span_end])
            global_image_feature = compressed[-1:]
            span_start = span_end

        tail_index = positions[span_start:]
        segments.append(
            self._fuse(
                inputs_embeds[tail_index],
                target_hidden_states[tail_index],
                global_image_feature,
            )
        )
        source_index.append(tail_index)
        return torch.cat(segments, dim=0), torch.cat(source_index, dim=0), global_image_feature

    def _prefill_generation(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, _DraftKVCache, torch.Tensor]:
        """Prefill the compressed ViSpec draft cache for one prompt.

        Args:
            inputs_embeds: Tensor of shape [1, sequence, hidden], shifted to the
                next-token alignment and ending with the speculative root embedding.
            target_hidden_states: Tensor of shape [1, sequence, hidden].
            attention_mask: Tensor of shape [1, sequence].
            image_mask: Bool tensor of shape [1, sequence], aligned with
                ``inputs_embeds``.

        Returns:
            Last hidden state of shape [1, hidden], per-layer K/V tensors of
            shape [1, kv_heads, compressed_sequence, head_dim], and the global
            image feature of shape [1, hidden].
        """
        if bool(image_mask.any()):
            hidden_states, source_index, global_image_feature = self._compress_sequence(
                inputs_embeds[0], target_hidden_states[0], image_mask[0].bool()
            )
            hidden_states = hidden_states.unsqueeze(0)
            position_ids = source_index.unsqueeze(0)
            compressed_attention_mask = attention_mask[:, source_index]
        else:
            global_image_feature = torch.zeros_like(inputs_embeds[0, :1])
            hidden_states = self._fuse(inputs_embeds[0], target_hidden_states[0], global_image_feature).unsqueeze(0)
            position_ids = torch.arange(
                inputs_embeds.shape[1], dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)
            compressed_attention_mask = attention_mask

        attention_bias = _build_causal_mask(compressed_attention_mask, hidden_states.dtype)
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden_states, layer_cache = layer._forward_cached(
                hidden_states,
                attention_bias,
                position_ids,
                None,
            )
            next_cache.append(layer_cache)
        hidden_states = self.norm(hidden_states)
        return hidden_states[:, -1], tuple(next_cache), global_image_feature

    def _decode_generation(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        global_image_feature: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: _DraftKVCache,
    ) -> tuple[torch.Tensor, _DraftKVCache]:
        """Decode cached text or tree nodes through the ViSpec draft.

        Args:
            inputs_embeds: Tensor of shape [1, query, hidden].
            target_hidden_states: Tensor of shape [1, query, hidden].
            global_image_feature: Tensor of shape [1, hidden].
            position_ids: Long tensor of shape [1, query].
            attention_mask: Additive tensor of shape [1, 1, query, cached + query].
            past_key_values: Per-layer K/V tensors of shape
                [1, kv_heads, cached, head_dim].

        Returns:
            Hidden states of shape [1, query, hidden] and per-layer K/V tensors
            of shape [1, kv_heads, cached + query, head_dim].
        """
        hidden_states = self._fuse(
            inputs_embeds[0],
            target_hidden_states[0],
            global_image_feature,
        ).unsqueeze(0)
        next_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer, layer_cache in zip(self.layers, past_key_values):
            hidden_states, next_layer_cache = layer._forward_cached(
                hidden_states,
                attention_mask,
                position_ids,
                layer_cache,
            )
            next_cache.append(next_layer_cache)
        return self.norm(hidden_states), tuple(next_cache)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next-position target hidden states from vision-aware features.

        Batch size must be 1: image spans have per-sample lengths, so the
        compressed sequences of two samples would not share a length. This
        matches the reference implementation, which raises on batch size > 1.

        Args:
            inputs_embeds: Tensor of shape [1, sequence, hidden] -- the target's
                embedding-layer output (vision features already spliced in),
                shifted left by one position so index ``i`` holds the embedding
                of token ``i + 1``.
            target_hidden_states: Tensor of shape [1, sequence, hidden] -- the
                target's last hidden state, *not* shifted.
            attention_mask: Tensor of shape [1, sequence]; 1 for real tokens,
                0 for padding.
            image_mask: Bool tensor of shape [1, sequence], aligned with
                ``inputs_embeds`` (True where token ``i + 1`` is an image token).

        Returns:
            Tensor of shape [1, sequence, hidden]. Positions consumed by the
            image compression carry zeros: they are never supervised, because
            the loss mask covers assistant-response positions only.
        """
        batch_size = inputs_embeds.shape[0]
        if batch_size != 1:
            raise NotImplementedError(
                f"ViSpec draft training runs at micro_batch_size=1 (got batch size {batch_size}); "
                "image spans differ per sample, so their compressed sequences have different lengths."
            )
        inputs_embeds = inputs_embeds.to(target_hidden_states.dtype)
        image_mask = image_mask.bool()

        if bool(image_mask.any()):
            hidden_states, source_index, _ = self._compress_sequence(
                inputs_embeds[0], target_hidden_states[0], image_mask[0]
            )
            hidden_states = hidden_states.unsqueeze(0)
            position_ids = source_index.unsqueeze(0)
            compressed_attention_mask = attention_mask[:, source_index]
        else:
            # Text-only sample: no span to compress, so ``img_adaptor`` would
            # receive no gradient and DDP (find_unused_parameters=False) would
            # error out on the rank that saw it. Keep it in the graph with a
            # zero-weighted term, exactly as the reference does.
            hidden_states = self._fuse(
                inputs_embeds[0],
                target_hidden_states[0],
                torch.zeros_like(inputs_embeds[0, :1]),
            ).unsqueeze(0)
            hidden_states = hidden_states + 0.0 * self.img_adaptor(inputs_embeds[:, :1]).sum(dim=1, keepdim=True)
            source_index = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = source_index.unsqueeze(0)
            compressed_attention_mask = attention_mask

        causal_mask = _build_causal_mask(compressed_attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)
        hidden_states = self.norm(hidden_states)

        # Scatter back to the original sequence layout so the caller can compare
        # against the (uncompressed) target supervision. Equivalent to the
        # reference's one-hot ``trans_mat`` einsum, without materializing a
        # [compressed_sequence, sequence] matrix.
        output = torch.zeros(
            (1, inputs_embeds.shape[1], hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return output.index_copy(1, source_index, hidden_states)


class VispecCachedTreeDraftGenerator:
    """Generate ViSpec trees with a persistent draft KV cache.

    Args:
        draft_model: ViSpec draft whose inference cache is retained across rounds.
        target_lm_head: Frozen target language-model head used to score draft features.
        target_embeddings: Frozen target token embedding module.
    """

    def __init__(self, draft_model: VispecDraftModel, target_lm_head: nn.Module, target_embeddings: nn.Module) -> None:
        self.draft_model = draft_model
        self.target_lm_head = target_lm_head
        self.target_embeddings = target_embeddings
        self._stable_cache: _DraftKVCache | None = None
        self._global_image_feature: torch.Tensor | None = None
        self._processed_target_tokens = 0

    def reset(self) -> None:
        """Drop the prompt-specific draft cache before starting another sample."""
        self._stable_cache = None
        self._global_image_feature = None
        self._processed_target_tokens = 0

    @staticmethod
    def _causal_decode_mask(
        *,
        query_length: int,
        cached_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive mask for cached linear draft decoding.

        Args:
            query_length: Number of new query tokens.
            cached_length: Number of existing draft KV entries.
            dtype: Floating-point mask dtype.
            device: Device on which to create the mask.

        Returns:
            Additive tensor of shape [1, 1, query, cached + query].
        """
        allowed = torch.ones((query_length, cached_length + query_length), dtype=torch.bool, device=device)
        allowed[:, cached_length:] = torch.tril(
            torch.ones((query_length, query_length), dtype=torch.bool, device=device)
        )
        mask = torch.zeros((1, 1, query_length, cached_length + query_length), dtype=dtype, device=device)
        return mask.masked_fill(~allowed.view(1, 1, query_length, -1), torch.finfo(dtype).min)

    @torch.inference_mode()
    def propose(
        self,
        *,
        shifted_inputs_embeds: torch.Tensor,
        input_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        shifted_image_mask: torch.Tensor,
        root_token_id: int,
        draft_steps: int,
        top_k: int,
        beam_width: int,
    ) -> MSDTreeProposal:
        """Create a tree using one cached draft forward per tree depth.

        Args:
            shifted_inputs_embeds: Tensor of shape [1, sequence, hidden], ending
                with the target-greedy root token embedding.
            input_hidden_states: Tensor of shape [1, sequence, hidden].
            attention_mask: Tensor of shape [1, sequence].
            shifted_image_mask: Bool tensor of shape [1, sequence].
            root_token_id: Target-greedy token after the cached prefix.
            draft_steps: Number of drafted tree levels after the root.
            top_k: Candidate count produced for each live parent.
            beam_width: Total number of non-root nodes retained for target
                verification. The live draft frontier retains ``top_k`` nodes.

        Returns:
            Flattened proposal with tree attention and retrieval tensors.
        """
        if draft_steps < 1 or top_k < 1 or beam_width < 1:
            raise ValueError("draft_steps, top_k, and beam_width must all be positive.")
        if shifted_inputs_embeds.shape != input_hidden_states.shape:
            raise ValueError("ViSpec cached draft inputs and hidden states must have matching shapes.")
        if shifted_inputs_embeds.shape[:2] != attention_mask.shape or attention_mask.shape != shifted_image_mask.shape:
            raise ValueError("ViSpec cached draft tensors must share matching [1, sequence] dimensions.")
        if shifted_inputs_embeds.shape[0] != 1:
            raise ValueError("ViSpec cached tree drafting requires batch size one.")

        root_ids = torch.tensor([[root_token_id]], dtype=torch.long, device=shifted_inputs_embeds.device)
        shifted_inputs_embeds = shifted_inputs_embeds.clone()
        shifted_inputs_embeds[:, -1:] = self.target_embeddings(root_ids).to(shifted_inputs_embeds.dtype)

        current_target_tokens = input_hidden_states.shape[1]
        if self._stable_cache is None:
            last_hidden, stable_cache, global_image_feature = self.draft_model._prefill_generation(
                shifted_inputs_embeds,
                input_hidden_states,
                attention_mask,
                shifted_image_mask,
            )
            self._stable_cache = stable_cache
            self._global_image_feature = global_image_feature
        else:
            new_tokens = current_target_tokens - self._processed_target_tokens
            if new_tokens < 1 or self._global_image_feature is None:
                raise RuntimeError("ViSpec draft cache did not receive a newly accepted target path.")
            cached_length = self._stable_cache[0][0].shape[-2]
            decode_mask = self._causal_decode_mask(
                query_length=new_tokens,
                cached_length=cached_length,
                dtype=input_hidden_states.dtype,
                device=input_hidden_states.device,
            )
            position_ids = torch.arange(
                self._processed_target_tokens,
                current_target_tokens,
                dtype=torch.long,
                device=input_hidden_states.device,
            ).unsqueeze(0)
            decoded, self._stable_cache = self.draft_model._decode_generation(
                shifted_inputs_embeds[:, -new_tokens:],
                input_hidden_states[:, -new_tokens:],
                self._global_image_feature,
                position_ids,
                decode_mask,
                self._stable_cache,
            )
            last_hidden = decoded[:, -1]
        self._processed_target_tokens = current_target_tokens

        root_logits = self.target_lm_head(last_hidden)
        root_log_probs = torch.log_softmax(root_logits, dim=-1)
        values, token_ids = torch.topk(root_log_probs, k=top_k, dim=-1)
        candidate_pool: list[_VispecDraftCandidate] = []
        frontier: list[_VispecDraftCandidate] = []
        next_candidate_index = 0
        for value, token_id in zip(values[0], token_ids[0]):
            token = int(token_id.item())
            score = float(value.item())
            candidate = _VispecDraftCandidate(next_candidate_index, -1, token, last_hidden[0], score, ())
            candidate_pool.append(candidate)
            frontier.append(candidate)
            next_candidate_index += 1

        temporary_cache = self._stable_cache
        stable_length = temporary_cache[0][0].shape[-2]
        for depth in range(1, draft_steps + 1):
            query_length = len(frontier)
            cached_length = temporary_cache[0][0].shape[-2]
            allowed = torch.zeros(
                (query_length, cached_length + query_length),
                dtype=torch.bool,
                device=input_hidden_states.device,
            )
            allowed[:, :stable_length] = True
            for row, candidate in enumerate(frontier):
                if candidate.ancestor_cache_indices:
                    allowed[row, torch.tensor(candidate.ancestor_cache_indices, device=allowed.device)] = True
                allowed[row, cached_length + row] = True
            tree_mask = torch.zeros(
                (1, 1, query_length, cached_length + query_length),
                dtype=input_hidden_states.dtype,
                device=input_hidden_states.device,
            ).masked_fill(~allowed.view(1, 1, query_length, -1), torch.finfo(input_hidden_states.dtype).min)
            frontier_ids = torch.tensor(
                [[candidate.token_id for candidate in frontier]],
                dtype=torch.long,
                device=input_hidden_states.device,
            )
            frontier_embeds = self.target_embeddings(frontier_ids).to(input_hidden_states.dtype)
            frontier_hidden = torch.stack([candidate.parent_hidden for candidate in frontier], dim=0).unsqueeze(0)
            position_ids = torch.full(
                (1, query_length),
                current_target_tokens + depth - 1,
                dtype=torch.long,
                device=input_hidden_states.device,
            )
            predicted_hidden, temporary_cache = self.draft_model._decode_generation(
                frontier_embeds,
                frontier_hidden,
                self._global_image_feature,
                position_ids,
                tree_mask,
                temporary_cache,
            )
            logits = self.target_lm_head(predicted_hidden[0])
            log_probs = torch.log_softmax(logits, dim=-1)
            child_values, child_ids = torch.topk(log_probs, k=top_k, dim=-1)
            candidates: list[_VispecDraftCandidate] = []
            for parent_row, parent in enumerate(frontier):
                physical_parent = cached_length + parent_row
                for child_rank in range(top_k):
                    candidate = _VispecDraftCandidate(
                        next_candidate_index,
                        parent.candidate_index,
                        int(child_ids[parent_row, child_rank].item()),
                        predicted_hidden[0, parent_row],
                        float(
                            (
                                child_values[parent_row, child_rank] + child_values.new_tensor(parent.log_probability)
                            ).item()
                        ),
                        (*parent.ancestor_cache_indices, physical_parent),
                    )
                    candidates.append(candidate)
                    candidate_pool.append(candidate)
                    next_candidate_index += 1
            candidates.sort(key=lambda candidate: candidate.log_probability, reverse=True)
            frontier = candidates[:top_k]

        selected_candidates = sorted(
            sorted(candidate_pool, key=lambda candidate: candidate.log_probability, reverse=True)[:beam_width],
            key=lambda candidate: candidate.candidate_index,
        )
        final_index_by_candidate = {
            candidate.candidate_index: final_index for final_index, candidate in enumerate(selected_candidates, start=1)
        }
        nodes = [
            MSDTreeNode(
                index=final_index_by_candidate[candidate.candidate_index],
                parent_index=0
                if candidate.parent_candidate_index < 0
                else final_index_by_candidate[candidate.parent_candidate_index],
                token_id=candidate.token_id,
                depth=self._candidate_depth(candidate, candidate_pool),
                log_probability=candidate.log_probability,
            )
            for candidate in selected_candidates
        ]

        parent_indices = {node.parent_index for node in nodes}
        leaf_indices = tuple(node.index for node in nodes if node.index not in parent_indices)
        layout = build_msd_tree_layout(nodes, leaf_indices, device=input_hidden_states.device)
        return MSDTreeProposal(root_token_id, tuple(nodes), leaf_indices, layout)

    @staticmethod
    def _candidate_depth(candidate: _VispecDraftCandidate, candidate_pool: list[_VispecDraftCandidate]) -> int:
        """Return one candidate's depth in the accumulated draft lattice.

        Args:
            candidate: Candidate whose ``parent_hidden`` is a Tensor of shape
                [hidden].
            candidate_pool: Candidates in generation order; each
                ``parent_hidden`` is a Tensor of shape [hidden].

        Returns:
            Candidate depth, with root children at depth one.
        """
        depth = 1
        parent_index = candidate.parent_candidate_index
        while parent_index >= 0:
            depth += 1
            parent_index = candidate_pool[parent_index].parent_candidate_index
        return depth
