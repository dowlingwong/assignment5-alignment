"""
CS336 Assignment 5: GRPO Training Script (Section 4.3 & 5.4 & 6.4)

Full GRPO training loop for OLMo-2-0425-1B on GSM8K.
Supports on-policy and off-policy variants.

Usage:
    python -m cs336_alignment.train_grpo \
        --model_id allenai/OLMo-2-0425-1B \
        --train_data data/gsm8k/train.jsonl \
        --val_data data/gsm8k/test.jsonl \
        --prompt r1_zero \
        --output_dir checkpoints/grpo_run1 \
        --seed 42
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer, save_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.prompting_baselines import load_gsm8k, parse_gsm8k_ground_truth, load_prompt_template, format_prompt


# ─────────────────────────────────────────────────────────
# Hyperparameters (suggested defaults from the assignment)
# ─────────────────────────────────────────────────────────
DEFAULT_HPARAMS = {
    "n_train_examples": 6400,
    "n_val_examples": 1024,
    "num_rollout_steps": 200,
    "learning_rate": 1e-5,
    "rollout_batch_size": 256,   # = train_batch_size; number of *responses*, not prompts
    "group_size": 8,
    "gradient_accumulation_steps": 32,
    "sampling_temperature": 1.0,
    "sampling_max_tokens": 512,
    "max_grad_norm": 1.0,
    "optimizer_betas": (0.9, 0.95),
    "weight_decay": 0.0,
    "val_every_n_rollout_steps": 10,
    "log_rollouts_every_n_steps": 40,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_val(
    vllm_server,
    val_prompts: list[str],
    val_ground_truths: list[str],
    reward_fn,
    sampling_params: dict,
    batch_size: int = 64,
) -> dict:
    """Run greedy decoding on the validation set and compute accuracy."""
    val_params = {**sampling_params, "temperature": 0.0, "top_p": 1.0}
    all_rewards = []
    all_format_rewards = []

    for start in range(0, len(val_prompts), batch_size):
        batch_prompts = val_prompts[start:start + batch_size]
        batch_gts = val_ground_truths[start:start + batch_size]
        completions = vllm_server.generate_completions(batch_prompts, val_params)
        for comp, gt in zip(completions, batch_gts):
            result = reward_fn(comp.text, gt)
            all_rewards.append(result["reward"])
            all_format_rewards.append(result["format_reward"])

    avg_len = 0.0
    if completions:
        all_lens = [len(vllm_server.tokenizer.encode(c.text)) for c in completions]
        avg_len = sum(all_lens) / len(all_lens)

    return {
        "val_reward": sum(all_rewards) / len(all_rewards),
        "val_format_reward": sum(all_format_rewards) / len(all_format_rewards),
        "val_avg_response_length": avg_len,
    }


def run_grpo_training(
    model_id: str,
    train_data_path: str,
    val_data_path: str,
    prompt_name: str,
    prompt_dir: str,
    output_dir: str,
    seed: int = 42,
    hparams: dict = None,
    # Algorithm variants
    baseline: str = "mean",
    advantage_normalizer: str = "std",
    loss_normalization: str = "sequence",
    normalization_constant: int | None = None,
    # Off-policy
    importance_reweighting_method: str = "none",
    cliprange: float | None = None,
    off_policy_steps: int = 1,
    # Devices
    train_gpu: int = 0,
    vllm_gpu: int = 1,
    # Logging
    use_wandb: bool = False,
    wandb_project: str = "cs336_grpo",
):
    """
    Main GRPO training loop.

    Algorithm (on-policy):
      For each rollout step:
        1. Sync training model weights → vLLM server
        2. Sample B prompts from training data
        3. Generate G responses per prompt with vLLM
        4. Compute rewards, normalize to advantages
        5. Take one policy gradient step (with gradient accumulation)
        6. Log metrics; periodically evaluate on validation set

    For off-policy (off_policy_steps > 1):
      Between syncs, take multiple gradient steps on the same rollout batch.
    """
    if hparams is None:
        hparams = DEFAULT_HPARAMS.copy()

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ── Import vLLM server ──────────────────────────────────────────
    from cs336_alignment.vllm_utils import VLLMServer

    # ── Load data ───────────────────────────────────────────────────
    train_data = load_gsm8k(train_data_path)[:hparams["n_train_examples"]]
    val_data = load_gsm8k(val_data_path)[:hparams["n_val_examples"]]
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Load prompt template ────────────────────────────────────────
    prompt_path = os.path.join(prompt_dir, f"{prompt_name}.prompt")
    prompt_template = load_prompt_template(prompt_path)

    if prompt_name == "question_only":
        reward_fn = question_only_reward_fn
        sampling_params = {
            "temperature": hparams["sampling_temperature"],
            "top_p": 1.0,
            "max_tokens": hparams["sampling_max_tokens"],
        }
    else:
        reward_fn = r1_zero_reward_fn
        sampling_params = {
            "temperature": hparams["sampling_temperature"],
            "top_p": 1.0,
            "max_tokens": hparams["sampling_max_tokens"],
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        }

    # ── Prepare val prompts ────────────────────────────────────────
    val_prompts = [format_prompt(prompt_template, ex["question"]) for ex in val_data]
    val_gts = [parse_gsm8k_ground_truth(ex["answer"]) for ex in val_data]

    # ── Load training model ────────────────────────────────────────
    train_device = f"cuda:{train_gpu}"
    print(f"Loading training model on {train_device}...")
    policy, tokenizer = get_model_and_tokenizer(model_id, device=train_device)
    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=hparams["learning_rate"],
        betas=hparams["optimizer_betas"],
        weight_decay=hparams["weight_decay"],
    )

    # ── Start vLLM server ──────────────────────────────────────────
    print(f"Starting vLLM server on GPU {vllm_gpu}...")
    vllm_server = VLLMServer(model_id=model_id, gpu=vllm_gpu)
    vllm_server.start()

    # ── WandB ──────────────────────────────────────────────────────
    if use_wandb:
        import wandb
        wandb.init(project=wandb_project, config={**hparams, "seed": seed, "prompt": prompt_name})

    # ── Training loop ──────────────────────────────────────────────
    group_size = hparams["group_size"]
    n_prompts_per_batch = hparams["rollout_batch_size"] // group_size
    all_metrics = []

    print(f"\nStarting GRPO training: {hparams['num_rollout_steps']} steps")
    print(f"  Prompts/batch: {n_prompts_per_batch}, Group size: {group_size}")
    print(f"  Variant: baseline={baseline}, normalizer={advantage_normalizer}, "
          f"loss_norm={loss_normalization}, reweighting={importance_reweighting_method}")

    for step in range(hparams["num_rollout_steps"]):
        step_start = time.time()

        # 1. Sync weights to vLLM
        vllm_server.sync_policy_weights(policy)

        # 2. Sample prompts
        batch_examples = random.choices(train_data, k=n_prompts_per_batch)
        batch_prompts = [format_prompt(prompt_template, ex["question"]) for ex in batch_examples]
        batch_gts = [parse_gsm8k_ground_truth(ex["answer"]) for ex in batch_examples]

        # Repeat prompts and ground_truths group_size times
        repeated_prompts = [p for p in batch_prompts for _ in range(group_size)]
        repeated_gts = [gt for gt in batch_gts for _ in range(group_size)]

        # 3. Generate rollouts with vLLM
        completions = vllm_server.generate_completions(
            repeated_prompts, sampling_params, batch_size=hparams["rollout_batch_size"]
        )
        rollout_responses = [c.text for c in completions]

        # 4. (Off-policy) compute old log-probs if needed
        old_log_probs = None
        if importance_reweighting_method != "none" and off_policy_steps > 1:
            # Compute log-probs under the current policy before any gradient step
            with torch.no_grad():
                policy.eval()
                batch_tokens = __import__("cs336_alignment.grpo", fromlist=["tokenize_prompt_and_output"]).tokenize_prompt_and_output(
                    repeated_prompts, rollout_responses, tokenizer
                )
                lp_result = __import__("cs336_alignment.grpo", fromlist=["get_response_log_probs"]).get_response_log_probs(
                    policy,
                    batch_tokens["input_ids"].to(train_device),
                    batch_tokens["labels"].to(train_device),
                )
                old_log_probs = lp_result["log_probs"].cpu()
                policy.train()

        # 5. Gradient step(s)
        n_grad_steps = off_policy_steps if importance_reweighting_method != "none" else 1
        for grad_step in range(n_grad_steps):
            loss, metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
                max_grad_norm=hparams["max_grad_norm"],
                reward_fn=reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=repeated_gts,
                group_size=group_size,
                baseline=baseline,
                advantage_normalizer=advantage_normalizer,
                loss_normalization=loss_normalization,
                normalization_constant=normalization_constant,
                importance_reweighting_method=importance_reweighting_method,
                old_log_probs=old_log_probs,
                cliprange=cliprange,
            )

        step_time = time.time() - step_start

        # 6. Logging
        log_dict = {
            "step": step,
            "loss": metadata.get("loss", 0.0),
            "grad_norm": metadata.get("grad_norm", 0.0),
            "token_entropy": metadata.get("token_entropy", 0.0),
            "train_reward": metadata.get("mean_reward", 0.0),
            "train_format_reward": metadata.get("mean_format_reward", 0.0),
            "step_time_s": step_time,
        }
        if "clip_fraction" in metadata:
            log_dict["clip_fraction"] = metadata["clip_fraction"]

        # 7. Validation
        if step % hparams["val_every_n_rollout_steps"] == 0 or step == hparams["num_rollout_steps"] - 1:
            val_metrics = evaluate_val(
                vllm_server, val_prompts, val_gts, reward_fn, sampling_params
            )
            log_dict.update(val_metrics)
            print(f"Step {step:4d} | loss={log_dict['loss']:.4f} | "
                  f"train_r={log_dict['train_reward']:.3f} | "
                  f"val_r={val_metrics['val_reward']:.3f} | "
                  f"entropy={log_dict['token_entropy']:.3f} | "
                  f"t={step_time:.1f}s")

        # 8. Log sample rollouts
        if step % hparams["log_rollouts_every_n_steps"] == 0:
            sample_idx = random.randint(0, len(repeated_prompts) - 1)
            print(f"\n  [Sample rollout @ step {step}]")
            print(f"  Prompt: {repeated_prompts[sample_idx][:100]}...")
            print(f"  Response: {rollout_responses[sample_idx][:200]}...")
            print(f"  Ground truth: {repeated_gts[sample_idx]}")
            print()

        if use_wandb:
            import wandb
            wandb.log(log_dict)

        all_metrics.append(log_dict)

    # ── Save model and metrics ────────────────────────────────────
    save_model_and_tokenizer(policy, tokenizer, output_dir)
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved model to {output_dir}")
    print(f"Saved metrics to {metrics_path}")

    if use_wandb:
        import wandb
        wandb.finish()

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for OLMo-2-0425-1B on GSM8K")
    parser.add_argument("--model_id", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--train_data", default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data", default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt", default="r1_zero",
                        choices=["question_only", "r1_zero", "r1_zero_three_shot_gsm8k"])
    parser.add_argument("--prompt_dir", default="cs336_alignment/prompts")
    parser.add_argument("--output_dir", default="checkpoints/grpo_run")
    parser.add_argument("--seed", type=int, default=42)
    # Algorithm variants
    parser.add_argument("--baseline", default="mean", choices=["mean", "none"])
    parser.add_argument("--advantage_normalizer", default="std", choices=["std", "none", "mean"])
    parser.add_argument("--loss_normalization", default="sequence", choices=["sequence", "constant"])
    parser.add_argument("--normalization_constant", type=int, default=None)
    # Off-policy
    parser.add_argument("--importance_reweighting", default="none",
                        choices=["none", "noclip", "grpo", "gspo"])
    parser.add_argument("--cliprange", type=float, default=None)
    parser.add_argument("--off_policy_steps", type=int, default=1)
    # Hyperparams
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    # GPU
    parser.add_argument("--train_gpu", type=int, default=0)
    parser.add_argument("--vllm_gpu", type=int, default=1)
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="cs336_grpo")

    args = parser.parse_args()

    hparams = DEFAULT_HPARAMS.copy()
    hparams["learning_rate"] = args.lr
    hparams["num_rollout_steps"] = args.num_steps
    hparams["group_size"] = args.group_size
    hparams["rollout_batch_size"] = args.rollout_batch_size

    run_grpo_training(
        model_id=args.model_id,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        prompt_name=args.prompt,
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        hparams=hparams,
        baseline=args.baseline,
        advantage_normalizer=args.advantage_normalizer,
        loss_normalization=args.loss_normalization,
        normalization_constant=args.normalization_constant,
        importance_reweighting_method=args.importance_reweighting,
        cliprange=args.cliprange,
        off_policy_steps=args.off_policy_steps,
        train_gpu=args.train_gpu,
        vllm_gpu=args.vllm_gpu,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
