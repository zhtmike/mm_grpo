# Configuration

MM-GRPO uses Hydra for configuration management, allowing flexible and composable training setups.

## Configuration System

MM-GRPO uses a hierarchical YAML configuration system based on Hydra. Configurations are organized into logical groups:

- **Algorithm**: RL algorithm settings
- **Data**: Dataset and data loading configuration
- **Actor**: Policy model configuration
- **Rollout**: Generation/inference configuration
- **Reward Model**: Reward function configuration
- **Trainer**: Training loop and logging configuration

## Configuration Files

Configuration files are located in `gerl/trainer/config/`:

```
gerl/trainer/config/
├── ppo_trainer.yaml          # Main configuration file
├── actor/
│   ├── actor.yaml            # Actor configuration
│   └── dp_actor.yaml          # Data parallel actor
├── data/
│   └── data.yaml             # Data configuration
├── rollout/
│   └── rollout.yaml          # Rollout configuration
├── reward_model/
│   ├── reward_model.yaml     # Reward model config
│   └── dp_reward_model.yaml  # Data parallel reward
└── ...
```

## Configuration Validation

Hydra validates configuration at startup. Common errors:

- **Missing Required Fields**: Check for `???` markers in configs
- **Type Mismatches**: Ensure YAML types match expected Python types
- **Invalid Paths**: Verify file paths exist and are accessible

