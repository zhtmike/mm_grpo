# API Reference

Complete API documentation for MM-GRPO. This section provides detailed documentation for all modules, classes, and functions.

## Overview

MM-GRPO is organized into several main modules:

- **Trainer**: Training logic and main entry points
- **Workers**: Distributed worker implementations
- **Utils**: Utility functions and helpers
- **Protocol**: Data structures and protocols

## Modules

### Trainer Module

The trainer module contains the main training logic:

- [`gerl.trainer.main_flowgrpo`](trainer.md#gerl.trainer.main_flowgrpo) - Main training entry point
- [`gerl.trainer.ppo.ray_diffusion_trainer`](trainer.md#gerl.trainer.ppo.ray_diffusion_trainer) - Ray-based PPO trainer
- [`gerl.trainer.ppo.core_algos`](trainer.md#gerl.trainer.ppo.core_algos) - Core RL algorithms
- [`gerl.trainer.ppo.reward`](trainer.md#gerl.trainer.ppo.reward) - Reward management

### Workers Module

The workers module contains distributed worker implementations:

- [`gerl.workers.actor`](workers.md#gerl.workers.actor) - Actor workers for policy updates
- [`gerl.workers.rollout`](workers.md#gerl.workers.rollout) - Rollout workers for generation
- [`gerl.workers.reward_manager`](workers.md#gerl.workers.reward_manager) - Reward computation workers
- [`gerl.workers.diffusers_model`](workers.md#gerl.workers.diffusers_model) - Diffusers model integration

### Utils Module

The utils module contains utility functions:

- [`gerl.utils.reward_score`](utils.md#gerl.utils.reward_score) - Reward scoring functions
- [`gerl.utils.dataset`](utils.md#gerl.utils.dataset) - Dataset utilities
- [`gerl.utils.checkpoint`](utils.md#gerl.utils.checkpoint) - Checkpoint management
- [`gerl.utils.lora`](utils.md#gerl.utils.lora) - LoRA utilities

### Protocol Module

The protocol module defines data structures:

- [`gerl.protocol`](protocol.md#gerl.protocol) - Data protocols and structures

## Quick Navigation

- [Trainer API](trainer.md) - Training classes and functions
- [Workers API](workers.md) - Worker implementations
- [Utils API](utils.md) - Utility functions
- [Protocol API](protocol.md) - Data structures

