# Training

This guide covers the training workflow in MM-GRPO, including setup, execution, and monitoring.

## Training Overview

MM-GRPO training follows a standard RL training loop:

1. **Rollout**: Generate images from prompts using the current policy
2. **Reward Computation**: Evaluate generated images using reward functions
3. **Policy Update**: Update the policy using PPO/GRPO algorithm
4. **Validation**: Periodically evaluate on validation set
5. **Checkpointing**: Save model checkpoints

## Monitoring Training

### WandB Integration

If configured (`trainer.logger='["console", "wandb"]'`):

- **Metrics**: Training and validation metrics
- **Samples**: Generated images (if `log_val_generations > 0`)
- **System**: GPU utilization, memory usage

Access dashboard at [wandb.ai](https://wandb.ai).
