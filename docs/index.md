# MM-GRPO

A fast and easy-to-use library to support RL training for multi-modal generative models, built on top of `verl`, `vLLM-Omni`, and `diffusers`.

## Key Features

- **Easy Integration**: Simple integration of diverse RL training algorithms for multi-modal generative models, including `FlowGRPO` and `FlowGRPO-Fast`
- **Scalable Training**: Efficient parallel training with an asynchronous streaming workflow
- **Model Compatibility**: Full compatibility with diffusion models from `diffusers`

## Supported Algorithms

- [x] [Flow-GRPO](https://arxiv.org/abs/2505.05470) - Online RL training for flow matching models
- [x] Flow-GRPO-Fast - Optimized version with improved sampling efficiency
- [ ] [Mix-GRPO](https://arxiv.org/html/2507.21802v1) (coming soon)
- [ ] [DiffusionNFT](https://arxiv.org/abs/2509.16117) (coming soon)

## Supported Async Strategies

- [x] Async Reward Computation during Rollout
- [x] [One-Step-Off Async Policy](https://arxiv.org/abs/2410.18252)

## Supported Models

- [x] [Stable-Diffusion-3.5](https://arxiv.org/abs/2403.03206)

## Supported Rewards

- [x] [PaddleOCR](https://arxiv.org/abs/2507.05595) - OCR-based reward for text rendering tasks
- [x] [QwenVL-OCR](https://arxiv.org/abs/2502.13923) - Vision-language model for OCR tasks
- [x] [UnifiedReward](https://arxiv.org/abs/2503.05236) - Unified reward model for various tasks

## Get Started

See the [Installation Guide](installation.md) and [Quick Start Guide](quickstart.md).

## Documentation Structure

- **[Installation](installation.md)**: Complete setup instructions
- **[Quick Start](quickstart.md)**: Get up and running quickly
- **[User Guide](user-guide/index.md)**: Comprehensive usage documentation
- **[API Reference](api-reference/index.md)**: Complete API documentation

## Citation

If you use MM-GRPO in your research, please cite:

```bibtex
@software{mm_grpo,
  title={MM-GRPO: Multi-Modal Generative Model RL Training},
  author={Leibniz CSI Research Lab},
  year={2025},
  url={https://github.com/leibniz-csi/mm_grpo}
}
```

## Acknowledgement

We appreciate the contributions of the following works:

- [verl](https://github.com/volcengine/verl) - Versatile RL framework
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni) - Efficient LLM serving
- [diffusers](https://github.com/huggingface/diffusers) - State-of-the-art diffusion models
- [Flow-GRPO](https://github.com/yifan123/flow_grpo) - Original Flow-GRPO implementation

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/leibniz-csi/mm_grpo/blob/main/LICENSE) file for details.

