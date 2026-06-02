"""
CS336 Assignment 5: GRPO Implementation
Group Relative Policy Optimization and variants for LLM reinforcement learning.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Problem: tokenize_prompt_and_output
# ---------------------------------------------------------------------------

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """
    Tokenize prompts and outputs separately, concatenate without special tokens,
    and build a response_mask aligned with labels.

    The key insight: we tokenize prompt and response separately (no special tokens),
    concatenate them, then build a mask that is 1 only for response tokens.
    We slice off the last token for input_ids and first token for labels
    (standard causal LM teacher-forcing setup).

    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of response strings.
        tokenizer: HuggingFace tokenizer.

    Returns:
        dict with keys:
          - "input_ids": (batch_size, max_len - 1) — prompt+response tokens, last sliced off
          - "labels":    (batch_size, max_len - 1) — same sequence shifted by 1
          - "response_mask": (batch_size, max_len - 1) — 1 where label token is a response token
    """
    batch_size = len(prompt_strs)
    assert len(output_strs) == batch_size

    # Tokenize each part separately, no special tokens
    prompt_ids_list = [
        tokenizer.encode(p, add_special_tokens=False) for p in prompt_strs
    ]
    output_ids_list = [
        tokenizer.encode(o, add_special_tokens=False) for o in output_strs
    ]

    # Concatenate prompt + output for each example
    combined_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    combined_lens = [len(ids) for ids in combined_ids_list]
    max_len = max(combined_lens)

    # Pad to max_len using tokenizer pad_token_id (or 0 if not set)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(combined_ids_list):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Standard causal LM split: input = all but last, labels = all but first
    input_ids_in = input_ids[:, :-1]   # (B, max_len-1)
    labels = input_ids[:, 1:]          # (B, max_len-1)

    # response_mask: 1 where the *label* token comes from the response
    # Label position t corresponds to predicting token at position t+1 in combined_ids.
    # The response starts at position len(prompt_ids) in combined_ids.
    # In labels (shifted by 1), response tokens start at index len(prompt_ids) - 1.
    response_mask = torch.zeros(batch_size, max_len - 1, dtype=torch.bool)
    for i, (p_ids, o_ids) in enumerate(zip(prompt_ids_list, output_ids_list)):
        resp_start = len(p_ids) - 1   # label index where response begins
        resp_end = len(p_ids) + len(o_ids) - 1  # exclusive
        response_mask[i, resp_start:resp_end] = True

    return {
        "input_ids": input_ids_in,
        "labels": labels,
        "response_mask": response_mask,
    }


# ---------------------------------------------------------------------------
# Problem: get_response_log_probs
# ---------------------------------------------------------------------------

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute per-token conditional log-probabilities log p_θ(y_t | x, y_{<t}).

    Uses the model in inference mode (no gradient computation) if called outside
    a training context. The caller is responsible for ensuring the correct mode.

    Args:
        model: HuggingFace causal LM (already on the correct device).
        input_ids: (batch_size, seq_len) — tokenized prompt+response (last token sliced off).
        labels: (batch_size, seq_len) — shifted token IDs.
        return_token_entropy: If True, also return per-token entropy.

    Returns:
        dict with:
          - "log_probs": (batch_size, seq_len) — log p(label_t | context)
          - "token_entropy": (batch_size, seq_len) — per-token entropy (only if requested)
    """
    # Forward pass: logits shape is (B, seq_len, vocab_size)
    logits = model(input_ids).logits  # (B, T, V)

    # log-softmax over vocabulary
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)

    # Gather the log-prob of the actual label at each position
    log_probs = log_probs_all.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        # H = -sum_v p_v * log p_v = -sum_v exp(log_p_v) * log_p_v
        probs = log_probs_all.exp()
        entropy = -(probs * log_probs_all).sum(dim=-1)  # (B, T)
        result["token_entropy"] = entropy

    return result


# ---------------------------------------------------------------------------
# Problem: compute_rollout_rewards
# ---------------------------------------------------------------------------

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute scalar rewards for each rollout response.

    Args:
        reward_fn: Takes (response, ground_truth) and returns dict with keys
                   "reward", "format_reward", "answer_reward".
        rollout_responses: List of model-generated responses (len = rollout_batch_size).
        repeated_ground_truths: Ground truths, repeated group_size times
                                (len = rollout_batch_size).

    Returns:
        (raw_rewards, metadata)
          raw_rewards: (rollout_batch_size,) float tensor
          metadata: dict with mean_reward, mean_format_reward, etc.
    """
    rewards_list = []
    format_rewards = []
    answer_rewards = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(response, gt)
        rewards_list.append(result["reward"])
        format_rewards.append(result.get("format_reward", 0.0))
        answer_rewards.append(result.get("answer_reward", 0.0))

    raw_rewards = torch.tensor(rewards_list, dtype=torch.float32)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "mean_format_reward": float(sum(format_rewards) / len(format_rewards)),
        "mean_answer_reward": float(sum(answer_rewards) / len(answer_rewards)),
    }

    return raw_rewards, metadata


# ---------------------------------------------------------------------------
# Problem: compute_group_normalized_rewards
# ---------------------------------------------------------------------------

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Normalize raw rewards within groups to produce advantages.

    Supports the following GRPO variants:
      - Standard GRPO: baseline="mean", advantage_normalizer="std"
      - Dr. GRPO:      baseline="mean", advantage_normalizer="none"
      - RFT:           baseline="none", advantage_normalizer="none"
      - MaxRL:         baseline="mean", advantage_normalizer="mean"

    Args:
        raw_rewards: (rollout_batch_size,) — flat rewards for all rollouts.
        group_size: Number of responses per prompt.
        baseline: "mean" subtracts per-group mean; "none" no baseline.
        advantage_eps: Small epsilon to avoid division by zero.
        advantage_normalizer: "std" divides by per-group std; "none" no division;
                              "mean" divides by per-group mean reward.

    Returns:
        (advantages, metadata)
          advantages: (rollout_batch_size,) normalized rewards
          metadata: dict with mean/std/max/min of rewards
    """
    n_prompts = len(raw_rewards) // group_size
    assert len(raw_rewards) == n_prompts * group_size

    # Reshape to (n_prompts, group_size)
    rewards_grouped = raw_rewards.view(n_prompts, group_size)

    # Compute group statistics
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)   # (n_prompts, 1)
    group_std = rewards_grouped.std(dim=1, keepdim=True)     # (n_prompts, 1) — uses Bessel's correction
    group_max = rewards_grouped.max(dim=1).values
    group_min = rewards_grouped.min(dim=1).values

    # Apply baseline subtraction
    if baseline == "mean":
        adjusted = rewards_grouped - group_mean
    elif baseline == "none":
        adjusted = rewards_grouped.clone()
    else:
        raise NotImplementedError(f"baseline={baseline!r} not supported")

    # Apply normalization
    if advantage_normalizer == "std":
        advantages = adjusted / (group_std + advantage_eps)
    elif advantage_normalizer == "none":
        advantages = adjusted
    elif advantage_normalizer == "mean":
        advantages = adjusted / (group_mean.abs() + advantage_eps)
    else:
        raise NotImplementedError(f"advantage_normalizer={advantage_normalizer!r} not supported")

    advantages = advantages.view(-1)  # flatten back to (rollout_batch_size,)

    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "group_std_mean": group_std.mean().item(),
    }

    return advantages, metadata


# ---------------------------------------------------------------------------
# Problem: compute_policy_gradient_loss
# ---------------------------------------------------------------------------

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute per-token policy-gradient loss.

    The loss is the NEGATIVE of the objective (so PyTorch gradient descent = gradient ascent).

    Supports:
      - "none":   On-policy GRPO (no importance reweighting)
      - "noclip": Token-level importance reweighting without clipping
      - "grpo":   PPO/GRPO-style clipped token-level reweighting
      - "gspo":   Sequence-level geometric mean importance weight with clipping

    Args:
        raw_rewards_or_advantages: (batch_size,) or (batch_size, 1) scalar advantage per rollout.
        policy_log_probs: (batch_size, seq_len) per-token log-probs under current policy.
        importance_reweighting_method: Which reweighting to apply.
        old_log_probs: (batch_size, seq_len) per-token log-probs under old/inference policy.
        cliprange: Clipping parameter ε for "grpo" and "gspo" methods.
        response_mask: (batch_size, seq_len) optional mask for "gspo" sequence-level log-ratio.

    Returns:
        (per_token_loss, metadata)
          per_token_loss: (batch_size, seq_len) — negative per-token policy gradient loss
          metadata: dict with clip fraction statistics
    """
    # Ensure advantages are (batch_size, 1) for broadcasting with (batch_size, seq_len)
    if raw_rewards_or_advantages.dim() == 1:
        advantages = raw_rewards_or_advantages.unsqueeze(1)  # (B, 1)
    else:
        advantages = raw_rewards_or_advantages  # (B, 1)

    metadata = {}

    if importance_reweighting_method == "none":
        # Standard on-policy: loss = -A * log π_θ(y_t | ...)
        per_token_loss = -advantages * policy_log_probs

    elif importance_reweighting_method == "noclip":
        # Token-level importance reweighting without clipping
        assert old_log_probs is not None, "old_log_probs required for noclip"
        log_ratio = policy_log_probs - old_log_probs  # (B, T)
        ratio = log_ratio.exp()
        per_token_loss = -advantages * ratio * policy_log_probs

    elif importance_reweighting_method == "grpo":
        # PPO/GRPO-style clipped token-level importance reweighting
        assert old_log_probs is not None and cliprange is not None
        log_ratio = policy_log_probs - old_log_probs  # (B, T)
        ratio = log_ratio.exp()                         # w_t = π_θ / π_0

        # Unclipped objective term
        unclipped = advantages * ratio

        # Clipped objective term: clip ratio to [1-ε, 1+ε]
        clipped_ratio = ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
        clipped = advantages * clipped_ratio

        # GRPO objective: min(unclipped, clipped)
        # mask function: for positive advantage, mask where ratio >= 1+ε
        #                for negative advantage, mask where ratio <= 1-ε
        per_token_loss = -torch.min(unclipped, clipped)

        # Track clip fraction
        clip_mask = (ratio > 1.0 + cliprange) | (ratio < 1.0 - cliprange)
        metadata["clip_fraction"] = clip_mask.float().mean()

    elif importance_reweighting_method == "gspo":
        # GSPO: sequence-level geometric mean importance weight
        assert old_log_probs is not None and cliprange is not None
        assert response_mask is not None, "response_mask required for gspo"

        log_ratio = policy_log_probs - old_log_probs  # (B, T)

        # Compute per-sequence geometric mean: exp(mean(log_ratio over response tokens))
        resp_mask_float = response_mask.float()
        seq_log_ratio = (log_ratio * resp_mask_float).sum(dim=1, keepdim=True) / (
            resp_mask_float.sum(dim=1, keepdim=True) + 1e-8
        )  # (B, 1)
        s = seq_log_ratio.exp()  # sequence-level importance weight (B, 1)

        # Clipped GSPO: min(A*s, A*clip(s, [1-ε, 1+ε]))
        unclipped = advantages * s
        clipped_s = s.clamp(1.0 - cliprange, 1.0 + cliprange)
        clipped = advantages * clipped_s

        # Broadcast over sequence length — same weight for all tokens in sequence
        # Take gradient through s which brings in per-token log_ratio contributions
        per_token_objective = torch.min(unclipped, clipped)  # (B, 1)

        # To get per-token loss that, when summed over tokens, gives correct gradient,
        # we distribute the sequence-level weight to each token via sequence normalization
        # The GSPO gradient is: A * s * (1/L) * sum_t ∇ log π(y_t)
        # So per-token loss = -A * s * log π(y_t)
        # But s must be treated as a constant w.r.t. π_θ (stop_grad on s in objective)
        # We reweight by the clipped/selected s value
        selected_s = torch.where(
            unclipped <= clipped,
            s.expand_as(policy_log_probs),
            clipped_s.expand_as(policy_log_probs),
        )
        per_token_loss = -advantages.expand_as(policy_log_probs) * selected_s.detach() * policy_log_probs

        clip_mask = (s > 1.0 + cliprange) | (s < 1.0 - cliprange)
        metadata["clip_fraction"] = clip_mask.float().mean()

    else:
        raise NotImplementedError(f"importance_reweighting_method={importance_reweighting_method!r}")

    return per_token_loss, metadata


# ---------------------------------------------------------------------------
# Problem: aggregate_loss_across_microbatch
# ---------------------------------------------------------------------------

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """
    Aggregate per-token policy gradient loss to a scalar.

    Two strategies:
      - "sequence": Average within each sequence over response tokens, then average
                    across sequences. Weights each sequence equally.
      - "constant": Sum all masked tokens, then divide by normalization_constant.
                    Used in Dr. GRPO, RFT, MaxRL (normalizer = B * G * L).

    Args:
        per_token_policy_gradient_loss: (batch_size, seq_len)
        mask: (batch_size, seq_len) — 1 for response tokens, 0 otherwise
        loss_normalization: "sequence" or "constant"
        normalization_constant: Required for "constant" mode.

    Returns:
        Scalar loss tensor (differentiable).
    """
    mask_float = mask.float()

    if loss_normalization == "sequence":
        # Average within each sequence, then average across sequences
        seq_lengths = mask_float.sum(dim=1).clamp(min=1.0)  # (B,)
        seq_loss = (per_token_policy_gradient_loss * mask_float).sum(dim=1) / seq_lengths  # (B,)
        return seq_loss.mean()

    elif loss_normalization == "constant":
        assert normalization_constant is not None and normalization_constant > 0
        total_loss = (per_token_policy_gradient_loss * mask_float).sum()
        return total_loss / normalization_constant

    else:
        raise NotImplementedError(f"loss_normalization={loss_normalization!r}")


# ---------------------------------------------------------------------------
# Problem: grpo_train_step
# ---------------------------------------------------------------------------

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """
    Execute one GRPO policy gradient update on a batch of rollouts.

    This function:
      1. Computes raw rewards for each rollout.
      2. Normalizes rewards to advantages within groups.
      3. Tokenizes prompts + responses.
      4. Splits the batch into gradient_accumulation_steps microbatches.
      5. For each microbatch: forward pass → compute loss → backward.
      6. (Optional) Clips gradient norm.
      7. Optimizer step + zero_grad.

    To correctly handle normalization across microbatches, the loss for each
    microbatch is scaled by (microbatch_size / total_batch_size) when using
    sequence normalization.

    Args:
        model: Policy model to train (on training device).
        tokenizer: Tokenizer.
        optimizer: Optimizer.
        gradient_accumulation_steps: Number of microbatches.
        max_grad_norm: If set, clip gradient norm before optimizer step.
        reward_fn: Reward function (response, gt) -> dict.
        repeated_prompts: Prompts repeated group_size times (len = rollout_batch_size).
        rollout_responses: Model-generated responses (len = rollout_batch_size).
        repeated_ground_truths: Ground truths repeated group_size times.
        group_size: Responses per prompt.
        baseline: Baseline strategy for advantage computation.
        advantage_eps: Small epsilon for normalization stability.
        advantage_normalizer: Normalizer for advantage computation.
        importance_reweighting_method: Off-policy reweighting method.
        old_log_probs: (rollout_batch_size, seq_len) — for off-policy methods.
        cliprange: Clip range ε for off-policy clipping.
        loss_normalization: "sequence" or "constant".
        normalization_constant: Required for "constant" normalization.

    Returns:
        (loss, metadata)
          loss: Scalar batch loss (for logging).
          metadata: Training statistics.
    """
    device = next(model.parameters()).device
    rollout_batch_size = len(rollout_responses)

    # 1. Compute raw rewards
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )

    # 2. Prune zero-advantage sequences to speed up training
    # Zero-advantage sequences contribute nothing to the gradient
    advantages, adv_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    # Identify non-zero advantage sequences
    nonzero_mask = advantages.abs() > 0
    if nonzero_mask.sum() == 0:
        # All advantages are zero — skip this batch
        optimizer.zero_grad()
        dummy_loss = torch.tensor(0.0, requires_grad=False)
        metadata = {**reward_metadata, **adv_metadata, "loss": 0.0, "grad_norm": 0.0}
        return dummy_loss, metadata

    # Filter to non-zero advantage sequences
    active_indices = nonzero_mask.nonzero(as_tuple=True)[0].tolist()
    active_prompts = [repeated_prompts[i] for i in active_indices]
    active_responses = [rollout_responses[i] for i in active_indices]
    active_advantages = advantages[active_indices]
    active_old_log_probs = old_log_probs[active_indices] if old_log_probs is not None else None

    n_active = len(active_indices)
    # Adjust gradient accumulation steps proportionally
    effective_grad_accum = max(1, round(gradient_accumulation_steps * n_active / rollout_batch_size))

    # 3. Tokenize all active (prompt, response) pairs at once
    batch_tokens = tokenize_prompt_and_output(active_prompts, active_responses, tokenizer)
    input_ids = batch_tokens["input_ids"].to(device)
    labels = batch_tokens["labels"].to(device)
    response_mask = batch_tokens["response_mask"].to(device)

    # Truncate old_log_probs to match tokenized length if needed
    if active_old_log_probs is not None:
        seq_len = input_ids.shape[1]
        if active_old_log_probs.shape[1] > seq_len:
            active_old_log_probs = active_old_log_probs[:, :seq_len]
        elif active_old_log_probs.shape[1] < seq_len:
            # Pad with zeros (will be masked out)
            pad = torch.zeros(
                n_active, seq_len - active_old_log_probs.shape[1],
                device=active_old_log_probs.device
            )
            active_old_log_probs = torch.cat([active_old_log_probs, pad], dim=1)
        active_old_log_probs = active_old_log_probs.to(device)

    # Compute normalization constant for "constant" mode (fixed: B*G*L)
    if loss_normalization == "constant" and normalization_constant is None:
        normalization_constant = rollout_batch_size * input_ids.shape[1]

    # 4. Split into microbatches and accumulate gradients
    microbatch_size = max(1, n_active // effective_grad_accum)
    total_loss = torch.tensor(0.0)
    all_metadata = {}
    total_entropy = 0.0
    n_entropy_tokens = 0

    model.train()
    optimizer.zero_grad()

    for start in range(0, n_active, microbatch_size):
        end = min(start + microbatch_size, n_active)
        mb_input_ids = input_ids[start:end]
        mb_labels = labels[start:end]
        mb_response_mask = response_mask[start:end]
        mb_advantages = active_advantages[start:end].to(device)
        mb_old_log_probs = active_old_log_probs[start:end] if active_old_log_probs is not None else None

        mb_size = end - start

        # Forward pass with gradients
        mb_logits = model(mb_input_ids).logits
        mb_log_probs_all = F.log_softmax(mb_logits, dim=-1)
        mb_log_probs = mb_log_probs_all.gather(
            dim=-1, index=mb_labels.unsqueeze(-1)
        ).squeeze(-1)  # (mb, T)

        # Per-token entropy for logging
        with torch.no_grad():
            mb_probs = mb_log_probs_all.exp()
            mb_entropy = -(mb_probs * mb_log_probs_all).sum(dim=-1)
            total_entropy += (mb_entropy * mb_response_mask.float()).sum().item()
            n_entropy_tokens += mb_response_mask.float().sum().item()

        # Policy gradient loss
        per_token_loss, loss_meta = compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_advantages,
            policy_log_probs=mb_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old_log_probs,
            cliprange=cliprange,
            response_mask=mb_response_mask,
        )

        # Aggregate loss
        if loss_normalization == "sequence":
            loss = aggregate_loss_across_microbatch(
                per_token_loss, mb_response_mask, loss_normalization="sequence"
            )
            # Scale by fraction of total batch in this microbatch
            loss = loss * (mb_size / n_active)
        else:
            loss = aggregate_loss_across_microbatch(
                per_token_loss, mb_response_mask,
                loss_normalization="constant",
                normalization_constant=normalization_constant,
            )
            # For constant normalization, loss already uses the global normalizer

        loss.backward()
        total_loss = total_loss + loss.detach().cpu()

        for k, v in loss_meta.items():
            if k not in all_metadata:
                all_metadata[k] = []
            all_metadata[k].append(v.item() if isinstance(v, torch.Tensor) else v)

    # 5. Clip gradient norm and take optimizer step
    grad_norm = None
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    else:
        # Compute grad norm for logging only
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = total_norm ** 0.5

    optimizer.step()
    optimizer.zero_grad()

    avg_token_entropy = total_entropy / max(n_entropy_tokens, 1)

    metadata = {
        **reward_metadata,
        **adv_metadata,
        "loss": total_loss.item(),
        "grad_norm": grad_norm,
        "token_entropy": avg_token_entropy,
        "n_active_sequences": n_active,
    }
    for k, vlist in all_metadata.items():
        metadata[k] = sum(vlist) / len(vlist)

    return total_loss, metadata
