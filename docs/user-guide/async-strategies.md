# Async Strategies

MM-GRPO supports asynchronous training strategies to improve training efficiency by overlapping computation and communication. By default, MM-GRPO uses a hybrid engine where actor, rollout, reference, and other components share the same resources.

## Overview

Asynchronous strategies allow different components of the training pipeline to run concurrently, reducing idle time and improving throughput. MM-GRPO supports three main async strategies:

- **Decoupled Actor and Rollout**: Separate resource pools for actor and rollout with async rollout
- **Rollout with Async Reward Computing**: Asynchronous batch reward computation during rollout generation
- **One-Step-Off Async Policy**: One-step-off asynchronous policy training with decoupled trainer and rollout

## Supported Strategies

- [x] [Decoupled Actor and Rollout](#decoupled-actor-and-rollout) - Support asynchronous rollout
- [x] [Rollout with Async Reward Computing](#rollout-with-async-reward-computing) - Asynchronous batch reward computing during rollout generation
- [x] [One-Step-Off Async Policy](#one-step-off-async-policy) - One-step-off asynchronous policy training with decoupled trainer and rollout

## Decoupled Actor and Rollout

!!! note "Not Used by Default"
    **Coupled actor and rollout is used by default.** This strategy decouples actor and rollout into standalone resource pools.

### Introduction

Decouple actor and rollout into standalone resource pools, and use async rollout. This allows for better resource allocation and parallelization.

### Configuration

To decouple actor and rollout into standalone resource pools and use async rollout, set config:

```bash
    actor_rollout_ref.hybrid_engine=False 
    actor_rollout_ref.rollout.mode="async" 
```

## Rollout with Async Reward Computing

!!! note "Used by Default"
    This strategy is **enabled by default** and significantly reduces rollout GPU idle time.

### Introduction

During the rollout generation loop, after generating responses for each micro batch, asynchronously launch reward computation for the current batch. By combining asynchronous reward computing with rollout generation, rollout's GPU idle time is significantly reduced.

**Visual Comparison:**

![Async Reward Computing](https://github.com/user-attachments/assets/430c3604-4af8-4e01-aabf-3ce2ac0648b0)

*Left: Synchronous reward computing. Right: Asynchronous reward computing during rollout.*

Reference: [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch/blob/main/scripts/train.py#L355).

### Configuration

The `with_reward` function is applied by default with the following config:

```bash
    actor_rollout_ref.rollout.with_reward=True 
```

### Performance

All experiments were conducted on *NVIDIA H800* GPUs using the OCR reward.

The following table shows the training throughput increase when using asynchronous reward computing (`with_reward=True`) compared to synchronous reward computing (`with_reward=False`):

| Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | `with_reward` | Batch Size | `rollout.n` | Training Samples/Step | `ppo_micro_batch_size_per_gpu` | Speedup (sec/step) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD3.5-M | Flow-GRPO-Fast | True | 1 | paddle-ocr | True | 8 | 16 | 8×16=128 | 8 | 21 | +5% |
| SD3.5-M | Flow-GRPO-Fast | True | 1 | qwenvl-ocr-vllm | True | 8 | 16 | 8×16=128 | 8 | 19 | +5% |
| SD3.5-M | Flow-GRPO-Fast | True | 8 | paddle-ocr | True | 64 | 16 | 64×16=1024 | 8 | 150 | +100% |

**Key Observations:**
- Small improvements (~5%) for single GPU setups
- Significant improvements (~100%) for multi-GPU setups
- More benefit when reward computation is slower (model-based rewards)

## One-Step-Off Async Policy

!!! note "Used by Default with Decoupled Setup"
    This strategy is **used by default when using decoupled actor and rollout**.

### Introduction

We support the one-step-off async trainer to parallelize the generation and training processes, utilizing samples generated in the previous step for the current training. It involves appropriately partitioning resources, allocating dedicated resources for rollout generation and actor training. By reducing resources allocated to the generation phase, GPU idle time during long-tail sample generation is mitigated. Throughout this process, generation and training parameters maintain a one-step off policy.

**Visual Comparison:**

![One-Step-Off Async Policy](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/docs/one_step_off_policy.png)

*Left: Synchronous training. Right: One-step-off asynchronous training*

### References

- [verl Recipe: One Step Off Policy Async Trainer](https://github.com/volcengine/verl/tree/main/recipe/one_step_off_policy)
- [Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models](https://arxiv.org/abs/2410.18252)

### Configuration

To apply `one-step-off` async strategy, set config:

```bash
    actor_rollout_ref.hybrid_engine=False
    actor_rollout_ref.async_strategy="one-step-off"
    actor_rollout_ref.rollout.mode="async"
```

### Performance

All experiments were conducted on *NVIDIA H800* GPUs using the OCR reward.

**Training GPU hours required to reach and maintain a validation reward score of approximately 0.8:**

| Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | Async Strategy | # GPUs for Actor | # GPUs for Rollout | Batch Size | `rollout.n` | Learning Rate | # Val Samples | Throughput | # GPU Hour | Script |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD3.5-M | Flow-GRPO-Fast | False | 3 | qwenvl-ocr-vllm* | one-step-off | 1 | 2 | 8 | 8 | 1e-4 | 32 | 1.07 | 2.04 | [run_sd3_fast_3p_a1_r2.sh](https://github.com/leibniz-csi/mm_grpo/blob/main/examples/flowgrpo_trainer/experimental/run_sd3_fast_3p_a1_r2.sh) |
| SD3.5-M | Flow-GRPO-Fast | False | 3 | qwenvl-ocr-vllm* | one-step-off | 2 | 1 | 16 | 8 | 1e-4 | 32 | 1.25 | 3.39 | [run_sd3_fast_3p_a2_r1.sh](https://github.com/leibniz-csi/mm_grpo/blob/main/examples/flowgrpo_trainer/experimental/run_sd3_fast_3p_a2_r1.sh) |
| SD3.5-M | Flow-GRPO-Fast | True | 3 | qwenvl-ocr-vllm* | - | 3 | 3 | 24 | 8 | 1e-4 | 33 | 1.42 | 3.06 | - |

**\*Note**: `UnifiedReward-Think-qwen3vl-32b` model was used in reward computing.

**Validation reward curve:**

![3p Comparison](https://github.com/user-attachments/assets/a9630a75-5cbf-48fe-996c-6c66a0b5f8be)

- `sd35_m_ocr_fast_3p_hybrid`: hybrid engine with 3 GPUs
- `sd35_m_ocr_fast_3p_a1_r2`: one-step-off async with 1 GPU for actor and 2 GPUs for rollout
- `sd35_m_ocr_fast_3p_a2_r1`: one-step-off async with 2 GPUs for actor and 1 GPU for rollout