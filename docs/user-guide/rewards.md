# Rewards

MM-GRPO supports multiple reward functions for evaluating generated images. This guide covers available rewards and how to use them.

## Available Rewards

### PaddleOCR

OCR-based reward using PaddleOCR for text recognition accuracy.

**Configuration**:
```yaml
data:
  reward_fn: ["paddle-ocr"]
```

### QwenVL-OCR

Vision-language model-based OCR reward using Qwen-VL model series via vLLM.

**Configuration**:
```yaml
data:
  reward_fn: ["qwenvl-ocr-vllm"]
```

### UnifiedReward

Unified reward model supporting multiple tasks and preferences.

**Configuration**:
```yaml
data:
  reward_fn: ["unified-reward-vllm"]
```

## Reward Usage

Typical steps to use a reward:

### 1. Install Dependencies and Launch Model Servers

#### Model-based or Rule-based Reward (CPU only)

For example, to use the PaddleOCR reward, install related dependencies:

```bash
pip install paddlepaddle "paddleocr>=3.0" python-Levenshtein
```

#### Model-based Reward with vLLM Online Serving

For rewards that use vLLM (QwenVL-OCR, UnifiedReward), you need to:

1. Install vLLM package:
   ```bash
   # vLLM installation, please refer to official installation for details
   uv pip install vllm
   ```
   Or follow the [official vLLM installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/).

2. Launch vLLM serving and set environment variables:

   To launch the vLLM server, environment variables of URL and model path should be set:

   | Reward | URL to access | Model to use |
   | --- | --- | --- |
   | QwenVL-OCR | `QWEN_VL_OCR_VLLM_URL` | `QWEN_VL_OCR_PATH` |
   | UnifiedReward | `UNIFIED_REWARD_VLLM_URL` | `UNIFIED_REWARD_PATH` |

   Note: See `QwenVLOCRVLLMScorer` and `UnifiedRewardVLLMScorer` in [vllm.py](https://github.com/leibniz-csi/mm_grpo/blob/main/gerl/utils/reward_score/vllm.py) for environment variable names and usage.

   **Example: Qwen2.5-VL-7B-Instruct for OCR**
   ```bash
   # Launch vLLM serving
   CUDA_VISIBLE_DEVICES=0 vllm serve ${CHECKPOINT_HOME}/Qwen/Qwen2.5-VL-7B-Instruct --host localhost --port 9529
   
   # Set access URL and model name
   export QWEN_VL_OCR_VLLM_URL=http://localhost:9529/v1
   export QWEN_VL_OCR_PATH=${CHECKPOINT_HOME}/Qwen/Qwen2.5-VL-7B-Instruct
   ```

   **Example: UnifiedReward-2.0-qwen3vl-32b**
   ```bash
   # Launch vLLM serving
   CUDA_VISIBLE_DEVICES=1,2,3,4 vllm serve ${CHECKPOINT_HOME}/CodeGoat24/UnifiedReward-2.0-qwen3vl-32b \
     --host localhost \
     --served-model-name UnifiedReward \
     --gpu-memory-utilization 0.9 \
     --tensor-parallel-size 4 \
     --port 8090
   
   # Set access URL and model name
   export UNIFIED_REWARD_VLLM_URL=http://localhost:8090/v1
   export UNIFIED_REWARD_PATH=UnifiedReward
   ```

### 2. Add Reward Names to Training Configuration

#### Single Reward

```bash
python3 -m gerl.trainer.main_flowgrpo \
    data.data_source=ocr \
    data.reward_fn='["paddle-ocr"]' \
    ...
```

#### Multiple Rewards

You can use different rewards for training and validation:

```bash
python3 -m gerl.trainer.main_flowgrpo \
    data.data_source=prompt \
    data.reward_fn='["qwenvl-ocr-vllm"]' \  # proxy reward to use for training
    data.val_reward_fn='["unified-reward-vllm", "qwenvl-ocr-vllm"]' \  # gold reward to use for validation set
    ...
```

**Notes**:
- If validation reward `val_reward_fn` is not set, it defaults to training reward `reward_fn`
- `data.data_source=prompt` is required when using multiple rewards
- Rewards are combined (currently averaged, weights may be added in future)

## Custom Reward Functions

MM-GRPO provides a framework for creating custom reward functions. Please refer to [Customize Reward Function](https://github.com/leibniz-csi/mm_grpo/tree/main/recipe/customize_reward) for detailed step-by-step instructions on how to customize reward scorers.

The customization process involves:

1. **Create Reward Scorer**: Implement a scorer class inheriting from `Scorer`
2. **Register Reward**: Add your scorer to `gerl/utils/reward_score/multi.py`
3. **Use in Configuration**: Reference your reward in training configuration
4. **Test**: Create unit tests before using in training

See the [recipe documentation](https://github.com/leibniz-csi/mm_grpo/tree/main/recipe/customize_reward/README.md) for complete examples including:
- Model-based or rule-based rewards (CPU only)
- API-serving model-based rewards (vLLM, SGLang)
