"""
CS336 Assignment 5: Prompting Baselines (Section 3)
Evaluates OLMo-2-0425-1B on GSM8K with three prompting strategies:
  - question_only
  - r1_zero (zero-shot chain-of-thought)
  - r1_zero_three_shot (3-shot chain-of-thought)
"""

import json
import re
from pathlib import Path

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn


def load_gsm8k(path: str) -> list[dict]:
    """Load GSM8K dataset from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def parse_gsm8k_ground_truth(answer_str: str) -> str:
    """
    GSM8K ground truths are formatted as: {rationale} #### {answer}
    Extract just the numeric answer after ####.
    """
    return answer_str.split("####")[-1].strip()


def load_prompt_template(prompt_path: str) -> str:
    """Load a prompt template from a text file."""
    with open(prompt_path) as f:
        return f.read()


def format_prompt(template: str, question: str) -> str:
    """Fill {question} placeholder in a prompt template."""
    return template.replace("{question}", question)


def evaluate_on_gsm8k(
    vllm_server,
    dataset: list[dict],
    prompt_template: str,
    reward_fn,
    sampling_params: dict,
    max_examples: int | None = None,
) -> dict:
    """
    Run model inference on GSM8K and compute reward statistics.

    Args:
        vllm_server: VLLMServer instance (already started).
        dataset: List of GSM8K examples with "question" and "answer" keys.
        prompt_template: Prompt template string with {question} placeholder.
        reward_fn: Function(response, ground_truth) -> dict with "reward",
                   "format_reward", "answer_reward".
        sampling_params: Dict of vLLM sampling parameters.
        max_examples: If set, evaluate on at most this many examples.

    Returns:
        dict with evaluation results and statistics.
    """
    if max_examples is not None:
        dataset = dataset[:max_examples]

    prompts = [format_prompt(prompt_template, ex["question"]) for ex in dataset]
    ground_truths = [parse_gsm8k_ground_truth(ex["answer"]) for ex in dataset]

    # Generate completions
    completions = vllm_server.generate_completions(prompts, sampling_params)

    results = []
    total_reward = 0.0
    total_format = 0.0
    total_answer = 0.0

    for ex, comp, gt in zip(dataset, completions, ground_truths):
        response_text = comp.text
        rewards = reward_fn(response_text, gt)
        results.append({
            "question": ex["question"],
            "ground_truth": gt,
            "response": response_text,
            "reward": rewards["reward"],
            "format_reward": rewards["format_reward"],
            "answer_reward": rewards["answer_reward"],
        })
        total_reward += rewards["reward"]
        total_format += rewards["format_reward"]
        total_answer += rewards["answer_reward"]

    n = len(results)
    stats = {
        "n_examples": n,
        "accuracy": total_reward / n,
        "format_reward": total_format / n,
        "answer_reward": total_answer / n,
        "results": results,
    }
    return stats


def categorize_outputs(results: list[dict]) -> dict:
    """
    Categorize model outputs into:
      1. Correct with both format=1 and answer=1
      2. Format=1 but answer=0
      3. Format=0 and answer=0
    """
    cat1 = [r for r in results if r["format_reward"] == 1 and r["answer_reward"] == 1]
    cat2 = [r for r in results if r["format_reward"] == 1 and r["answer_reward"] == 0]
    cat3 = [r for r in results if r["format_reward"] == 0 and r["answer_reward"] == 0]
    return {
        "correct_format_and_answer": cat1,
        "correct_format_wrong_answer": cat2,
        "wrong_format": cat3,
    }


if __name__ == "__main__":
    """
    Example usage — requires a running GPU environment with vLLM.
    This script is illustrative; adapt paths and parameters for your setup.
    """
    import sys
    import os

    # Add the repo root to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from cs336_alignment.vllm_utils import VLLMServer

    MODEL_ID = "allenai/OLMo-2-0425-1B"
    DATA_DIR = Path(__file__).parent.parent / "data" / "gsm8k"
    PROMPT_DIR = Path(__file__).parent.parent / "cs336_alignment" / "prompts"

    # Load dataset
    test_data = load_gsm8k(str(DATA_DIR / "test.jsonl"))
    print(f"Loaded {len(test_data)} GSM8K test examples")

    # Load prompt templates
    question_only_template = load_prompt_template(str(PROMPT_DIR / "question_only.prompt"))
    r1_zero_template = load_prompt_template(str(PROMPT_DIR / "r1_zero.prompt"))
    r1_zero_3shot_template = load_prompt_template(str(PROMPT_DIR / "r1_zero_three_shot_gsm8k.prompt"))

    # Start vLLM server
    server = VLLMServer(model_id=MODEL_ID, gpu=0)
    server.start()

    # Sampling parameters
    base_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
    }
    r1_params = {
        **base_params,
        "stop": ["</answer>"],
        "include_stop_str_in_output": True,
    }

    prompts_and_configs = [
        ("question_only", question_only_template, question_only_reward_fn, base_params),
        ("r1_zero", r1_zero_template, r1_zero_reward_fn, r1_params),
        ("r1_zero_three_shot", r1_zero_3shot_template, r1_zero_reward_fn, r1_params),
    ]

    for name, template, reward_fn, params in prompts_and_configs:
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print('='*50)

        stats = evaluate_on_gsm8k(
            vllm_server=server,
            dataset=test_data,
            prompt_template=template,
            reward_fn=reward_fn,
            sampling_params=params,
            max_examples=200,  # Use subset for speed; remove for full eval
        )

        categories = categorize_outputs(stats["results"])
        print(f"Accuracy: {stats['accuracy']:.3f}")
        print(f"Format reward: {stats['format_reward']:.3f}")
        print(f"Answer reward: {stats['answer_reward']:.3f}")
        print(f"\nCategory breakdown:")
        print(f"  (1) Format=1, Answer=1: {len(categories['correct_format_and_answer'])}")
        print(f"  (2) Format=1, Answer=0: {len(categories['correct_format_wrong_answer'])}")
        print(f"  (3) Format=0, Answer=0: {len(categories['wrong_format'])}")

        # Show 3 examples from each category
        for cat_name, cat_items in categories.items():
            if cat_items:
                example = cat_items[0]
                print(f"\n--- Example from {cat_name} ---")
                print(f"Q: {example['question'][:100]}...")
                print(f"GT: {example['ground_truth']}")
                print(f"Response (first 200 chars): {example['response'][:200]}...")

        # Save results
        with open(f"gsm8k_{name}_results.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved to gsm8k_{name}_results.json")
