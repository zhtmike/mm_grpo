# Algorithms

MM-GRPO supports multiple RL algorithms for training flow matching models. This page describes the available algorithms and their characteristics. More algorithms will be added along the way!

## Flow-GRPO

Flow-GRPO is the first method to integrate online policy gradient reinforcement learning into flow matching models.

### Key Features

- **ODE-to-SDE Conversion**: Transforms deterministic ODEs into equivalent SDEs that match the original model's marginal distribution at all timesteps
- **Statistical Sampling**: Enables exploration for RL training
- **Online Learning**: Updates policy during training using collected rollouts

## Flow-GRPO-Fast

Flow-GRPO-Fast is an optimized version that improves sampling efficiency without sacrificing performance.

### Key Features

- **Denoising Reduction**: Reduces training denoising steps while retaining inference steps
- **SDE Windowing**: Uses windowed SDE sampling for efficiency

## References

- [Flow-GRPO Paper](https://arxiv.org/abs/2505.05470)
- [Original Flow-GRPO Repository](https://github.com/yifan123/flow_grpo)

