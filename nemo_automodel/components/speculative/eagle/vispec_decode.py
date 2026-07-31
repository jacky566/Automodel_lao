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

"""Tree-attention greedy ViSpec decoding for batch-one Transformers targets."""

from __future__ import annotations

import torch

from nemo_automodel.components.speculative.eagle.msd_decode import (
    MSDTreeProposal,
    MSDVerificationResult,
)
from nemo_automodel.components.speculative.eagle.target_v12 import _shift_left_with_zero
from nemo_automodel.components.speculative.eagle.vispec_draft import VispecCachedTreeDraftGenerator, VispecDraftModel
from nemo_automodel.components.speculative.eagle.vispec_target import (
    HFVispecTargetModel,
    VispecGenerationState,
)


class VispecCachedGreedyDecoder:
    """Draft and verify each ViSpec tree with one target KV-cache forward."""

    def __init__(self, target: HFVispecTargetModel, draft_model: VispecDraftModel) -> None:
        self.target = target
        self.draft_model = draft_model
        self.generator = VispecCachedTreeDraftGenerator(
            draft_model,
            target.get_lm_head(),
            target.get_input_embeddings(),
        )
        self.state: VispecGenerationState | None = None

    def prefill(self, model_inputs: dict[str, torch.Tensor]) -> None:
        """Reset the decoder with one multimodal prompt.

        Args:
            model_inputs: Processor output containing ``input_ids`` and
                ``attention_mask`` tensors of shape [1, sequence], with vision
                tensor layouts documented by
                :meth:`HFVispecTargetModel.prefill_generation`.
        """
        self.state = self.target.prefill_generation(model_inputs)
        self.generator.reset()

    def _verify(
        self,
        proposal: MSDTreeProposal,
        tree_logits: torch.Tensor,
    ) -> tuple[MSDVerificationResult, torch.Tensor]:
        """Select the longest target-greedy path from flattened tree logits.

        Args:
            proposal: Draft tree whose candidate paths are tuples of token ids.
            tree_logits: Tensor of shape [1, tree, vocab], where each tree node
                predicts the token at its child position.

        Returns:
            Verification result and a tensor of shape [accepted] containing the
            accepted root-to-leaf tree-node indices.
        """
        if self.state is None:
            raise RuntimeError("Call prefill() before ViSpec decoding.")
        best_accept_length = -1
        best_leaf_index: int | None = None
        best_path: tuple[int, ...] = ()
        best_tree_indices: torch.Tensor | None = None
        best_bonus_logits: torch.Tensor | None = None
        tree_token_ids = torch.tensor(
            [[proposal.root_token_id, *(node.token_id for node in proposal.nodes)]],
            dtype=self.state.input_ids.dtype,
            device=self.state.input_ids.device,
        )
        for leaf_index, path, padded_tree_indices in zip(
            proposal.leaf_indices,
            proposal.candidate_paths(),
            proposal.layout.retrieve_indices,
        ):
            tree_indices = padded_tree_indices[padded_tree_indices.ge(0)]
            path_logits = torch.cat(
                (
                    self.state.next_token_logits.unsqueeze(1),
                    tree_logits.index_select(1, tree_indices[:-1]),
                ),
                dim=1,
            )
            candidate_ids = tree_token_ids.index_select(1, tree_indices)
            matches = candidate_ids.eq(path_logits.argmax(dim=-1))
            accept_length = int(matches.cumprod(dim=1).sum().item())
            if accept_length > best_accept_length:
                best_accept_length = accept_length
                best_leaf_index = leaf_index
                best_path = path
                best_tree_indices = tree_indices[:accept_length]
                if accept_length > 0:
                    best_bonus_logits = tree_logits[:, tree_indices[accept_length - 1]]
                else:
                    best_bonus_logits = self.state.next_token_logits

        if best_tree_indices is None or best_bonus_logits is None or best_accept_length < 1:
            raise RuntimeError("The target-greedy ViSpec tree root must always be accepted.")

        return (
            MSDVerificationResult(
                accepted_token_ids=best_path[:best_accept_length],
                bonus_token_id=int(best_bonus_logits.argmax(dim=-1).item()),
                accepted_draft_tokens=max(0, best_accept_length - 1),
                leaf_index=best_leaf_index,
            ),
            best_tree_indices,
        )

    @torch.inference_mode()
    def decode_round(
        self,
        *,
        draft_steps: int,
        top_k: int,
        beam_width: int,
    ) -> tuple[MSDTreeProposal, MSDVerificationResult]:
        """Draft, verify, and commit one speculative round to the target cache.

        Args:
            draft_steps: Number of recursive draft expansion steps.
            top_k: Candidate count retained from each draft distribution.
            beam_width: Total number of non-root draft nodes retained in the
                target verification tree.

        Returns:
            Draft proposal and its target-greedy verification result.
        """
        if self.state is None:
            raise RuntimeError("Call prefill() before ViSpec decoding.")
        state = self.state
        root_token_id = int(state.next_token_logits.argmax(dim=-1).item())
        proposal = self.generator.propose(
            shifted_inputs_embeds=_shift_left_with_zero(state.inputs_embeds),
            input_hidden_states=state.input_hidden_states,
            attention_mask=state.attention_mask,
            shifted_image_mask=_shift_left_with_zero(state.image_mask),
            root_token_id=root_token_id,
            draft_steps=draft_steps,
            top_k=top_k,
            beam_width=beam_width,
        )
        tree_token_ids = torch.tensor(
            [[proposal.root_token_id, *(node.token_id for node in proposal.nodes)]],
            dtype=state.input_ids.dtype,
            device=state.input_ids.device,
        )
        tree_output = self.target.forward_tree_generation(
            state,
            token_ids=tree_token_ids,
            tree_attention_mask=proposal.layout.attention_mask,
            tree_position_ids=proposal.layout.position_ids,
        )
        result, accepted_tree_indices = self._verify(proposal, tree_output.logits)
        self.state = self.target.commit_tree_generation(
            state,
            tree_output,
            accepted_tree_indices=accepted_tree_indices,
        )
        return proposal, result
